#!/bin/bash

sudo fuser -v /dev/nvidia*
nvidia-smi
sudo kill -9 PID