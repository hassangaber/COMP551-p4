#!/bin/bash

sudo git clone https://github.com/casperhansen/fake-news-reasoning
cd fake-news-reasoning

sudo wget https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip?dl=1

cd code-acl/bias
sudo pip3 install -r requirements.txt
