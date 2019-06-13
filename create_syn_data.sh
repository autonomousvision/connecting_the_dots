#!/bin/bash

cd data/lcn
python setup.py build_ext --inplace

cd ../
python create_syn_data.py
