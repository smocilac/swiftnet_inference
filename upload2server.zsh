#!/bin/zsh

SCP_IP=
SCP_USER=
SCP_PATH=

scp -r $(pwd)/common         ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar
scp -r $(pwd)/tclap          ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar
scp    $(pwd)/CMakeLists.txt ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar
scp    $(pwd)/main.cpp       ${SCP_USER}@${SCP_IP}:${SCP_PATH}/dipl_seminar




