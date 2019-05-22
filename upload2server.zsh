#!/bin/zsh

SCP_IP=
SCP_USER=
SCP_PATH=/home/

scp -r /home/stjepan/Faks/dipl_seminar/common ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar
scp -r /home/stjepan/Faks/dipl_seminar/CMakeLists.txt ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar
scp -r /home/stjepan/Faks/dipl_seminar/main.cpp ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar



