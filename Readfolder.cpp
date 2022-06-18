#include <stdio.h>
#include <io.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include<fstream>  //ifstream
#include<string>     //包含getline()
#include<cmath>
using namespace std;

//get the name of document under a certain path
void getFileName(string path, vector<string>& files)
{
    long hFile = 0;
    struct _finddata_t fileinfo;
    string pathname = path + "\\*";
    if ((hFile = _findfirst(pathname.c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    getFileName(path + "\\" + fileinfo.name, files);
                }
            }
            else
            {
                files.push_back(path + "\\" + fileinfo.name);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}


int main(void) {
    intptr_t Handle;
    struct _finddata_t FileInfo;
    string p;
    string path = ".\\Testing";
    vector<string> files_path;

    getFileName(path, files_path);
    // delete the first two elements of files_path
    //files_path.erase(files_path.begin());
    //files_path.erase(files_path.begin());

    //




    return 0;
}