import numpy as np
import pandas as pd
import sys
import os


def read_data_file(data_file, data_folder="../"):
    """ Read data file into pandas data frame
    :type data_file:string
    :param data_file:name of file containing data in csv format

    :type data_folder:string
    :param data_folder:absolute path path of folder where files are located,
                       must be within parent directory

    :raises:none

    :rtype:pandas data frame
        Contains read data with names (if available)
    """
    dt_frame = pd.read_csv(data_folder+data_file)
    return dt_frame


def main():
    dt_frame = read_data_file("iris.data", """F:/Excelsior/PyML/code/data/iris/""" )
    print(dt_frame)

if __name__ == '__main__':
    main()
