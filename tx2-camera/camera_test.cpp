#include <opencv2/opencv.hpp>
#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <memory>
#include <unistd.h>
#include <vector>

#include "cudaNormalize.h"
#include "cudaFont.h"
#include "cudaMappedMemory.h"
#include "cudaOverlay.h"
#include "cudaResize.h"

#include "TrtNet.h"
#include "YoloLayer.h"


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
		
		
		
bool signal_recieved = false;
float* mClassColors[2];

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

bool DrawBoxes( float* input, float* output, uint32_t width, uint32_t height, const float* boundingBoxes, int numBoxes, int classIndex )
{
	if( !input || !output || width == 0 || height == 0 || !boundingBoxes || numBoxes < 1 || classIndex < 0 )
		return false;


    //const float4 bbox_ = make_float4(output[0], output[1], std::min((uint32_t)output[2], width), std::min((uint32_t)output[3], height));
	
	
	const float4 color = make_float4( mClassColors[0][classIndex*4+0] , 
									  mClassColors[0][classIndex*4+1],
									  mClassColors[0][classIndex*4+2],
									  mClassColors[0][classIndex*4+3] );
	
	printf("draw boxes  %i  %i \n", width, height);
	printf("draw boxes  %i  %i   %f %f %f %f\n", numBoxes, classIndex, color.x, color.y, color.z, color.w);
    //printf("draw boxes  %f %f %f %f\n", boundingBoxes[0], boundingBoxes[1], boundingBoxes[2], boundingBoxes[3]);
	
    if( CUDA_FAILED(cudaRectOutlineOverlay((float4*)input, (float4*)output, width, height, (float4*)boundingBoxes, numBoxes, color)) )
		return false;
	
	return true;
}

std::vector<float> prepareImage(cv::Mat& rgb)
{
    using namespace cv;

    int c = 3;
    int h = 608;   //net h
    int w = 608;   //net w

    float scale = min(float(w)/rgb.cols,float(h)/rgb.rows);
    auto scaleSize = cv::Size(rgb.cols * scale,rgb.rows * scale);

    // cv::Mat rgb ;
    // cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(c);
    cv::split(img_float, input_channels);

    std::vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
	// cudaMemcpy(data, input_channels[i].data, channelLength*sizeof(float), cudaMemcpyDeviceToDevice);
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}


int main( int argc, char** argv )
{
	int classNum = 80;

	cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], classNum * sizeof(float4)); 

	for( uint32_t n=0; n < classNum; n++ )
	{
		if( n != 1 )
		{
			mClassColors[0][n*4+0] = 0.0f;	// r
			mClassColors[0][n*4+1] = 200.0f;	// g
			mClassColors[0][n*4+2] = 255.0f;	// b
			mClassColors[0][n*4+3] = 100.0f;	// a
		}
		else
		{
			mClassColors[0][n*4+0] = 0.0f;	// r
			mClassColors[0][n*4+1] = 255.0f;	// g
			mClassColors[0][n*4+2] = 175.0f;	// b
			mClassColors[0][n*4+3] = 100.0f;	// a
		}
	}


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\nyolov3-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\nyolov3-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	
	/*
	 * Create yolov3 net
	 */
	std::string saveName = "/home/nvidia/models/yolov3_fp16.engine";
	Tn::trtNet net(saveName);

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\nyolov3-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("yolov3-camera:  failed to create openGL texture\n");
	}
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nyolov3-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nyolov3-camera:  camera open for streaming\n");
	

    /*
     * allocate memory for output bounding boxes and class confidence
     */
    int outputCount = net.getOutputSize() / sizeof(float);
	
    float* outdata = NULL;
    float* outdataCUDA = NULL;
	
    if( !cudaAllocMapped((void**)&outdata, (void**)&outdataCUDA, outputCount * sizeof(float)) ){
        printf("yolov3-camera:  failed to alloc output memory\n");
        return 0;
    }


    /*
     * processing loop
     */
    int i_iter;
    Yolo::Detection *output_ptr;
	int frame_counter;

	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
        if( !camera->Capture(&imgCPU, &imgCUDA, 100000) )
            printf("\nyolov3-camera:  failed to capture frame\n");
        else
        {
            ++frame_counter;
            //printf("yolov3-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
        }

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA))
			printf("yolov3-camera:  failed to convert from NV12 to RGBA\n");
		
		cv::Mat imgNV12_mat(camera->GetHeight(), camera->GetWidth(), CV_8U, imgCUDA);
		cv::Mat rgbImage(camera->GetHeight(), camera->GetWidth(), CV_8UC3);
		cv::cvtColor(imgNV12_mat, rgbImage, 90);
		
		std::vector<float> preprocessed_input = prepareImage(rgbImage);
		printf("prepr-sz = %d\n", preprocessed_input.size());
        // sleep(1);

		if (preprocessed_input.size() <= 0){
			printf("yolov3-camera: Could not preprocess image.\n");
            break;
		}

		net.doInference(preprocessed_input.data(), outdata);
		//net.doInference(imgRGBA, outdata.get());

        float *output = (float *)outdata;
        auto outputCUDA = (float *) ((size_t) outdataCUDA);
		int count = output[0];
		printf("%d\n", count);
        output_ptr = (Yolo::Detection *) &output[1];


//        for (i_iter = 0; i_iter < count; ++i_iter)
//        {
//            printf("yolov3-camera: frame %d ; class %d ; probability %f \n", frame_counter, output_ptr[i_iter].classId, output_ptr[i_iter].prob);
//            printf("               x = %f ; y = %f ; w = %04.1f ; h = %04.1f \n", output_ptr[i_iter].bbox[0], output_ptr[i_iter].bbox[1], output_ptr[i_iter].bbox[2], output_ptr[i_iter].bbox[3]);

//            output_ptr[i_iter].bbox[0] *= camera->GetWidth();
//            output_ptr[i_iter].bbox[1] *= camera->GetHeight();
//            output_ptr[i_iter].bbox[2] += output_ptr[i_iter].bbox[0];
//            output_ptr[i_iter].bbox[3] += output_ptr[i_iter].bbox[1];

//            // This is a hack because cuda float4 type requires memory alignment to 16 bytes
//            // if (address & 0x0000000f) // MUST BE ZERO
//            //     return error;
//            // TODO rewrite logic for this so that memory is aligned
//            output[i_iter*4+0] = output_ptr[i_iter].bbox[0] ;
//            output[i_iter*4+1] = output_ptr[i_iter].bbox[1] ;
//            output[i_iter*4+2] = output_ptr[i_iter].bbox[2] ;
//            output[i_iter*4+3] = output_ptr[i_iter].bbox[3] ;
//        }
//        CUDA(cudaDeviceSynchronize());

		
        for (i_iter = 0; i_iter < count; ++i_iter)
		{
            printf("yolov3-camera: frame %d ; class %d ; probability %f \n", frame_counter, output_ptr[i_iter].classId, output_ptr[i_iter].prob);
            printf("               x = %f ; y = %f ; w = %04.1f ; h = %04.1f \n", output_ptr[i_iter].bbox[0], output_ptr[i_iter].bbox[1], output_ptr[i_iter].bbox[2], output_ptr[i_iter].bbox[3]);
            output_ptr[i_iter].bbox[0] *= camera->GetWidth();
            output_ptr[i_iter].bbox[1] *= camera->GetHeight();
            output_ptr[i_iter].bbox[2] += output_ptr[i_iter].bbox[0];
            output_ptr[i_iter].bbox[3] += output_ptr[i_iter].bbox[1];

            // This is a hack because cuda float4 type requires memory alignment to 16 bytes
            // if (address & 0x0000000f) // MUST BE ZERO
            //     return error;
            // TODO rewrite logic for this so that memory is aligned
            output[0] = output_ptr[i_iter].bbox[0] ;
            output[1] = output_ptr[i_iter].bbox[1] ;
            output[2] = output_ptr[i_iter].bbox[2] ;
            output[3] = output_ptr[i_iter].bbox[3] ;

			
            if( !DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
                                        outputCUDA + i_iter*4, 1, output_ptr[i_iter].classId) ) {
				printf("yolov3-camera:  failed to draw boxes\n");
				exit(0);			
			}
            CUDA(cudaDeviceSynchronize());
            /*
            if( font != NULL && i_iter == 0)
			{
				char str[256];
				sprintf(str, "class %d ; probability %f \n", output_ptr[i_iter].classId, output_ptr[i_iter].prob);
	
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
            }*/

            if( display != NULL )
            {
                char str[256];
                sprintf(str, "TensorRT %04.1f FPS", display->GetFPS());
                display->SetTitle(str);
            }
            break;

		}	

		

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\nyolov3-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("yolov3-camera:  video device has been un-initialized.\n");
	return 0;
}

