#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "config.h"

using namespace std;
void TransEFS(const string &dataname, map<string, size_t> &MapParameter);

void CheckDataset(const string &dataname, map<string, size_t> &MapParameter, map<string, string> &MapString){

    size_t data_size_millions = MapParameter["data_size_millions"];
    string path_dataset = "../dataset/" + dataname + "/";
    MapString["dataname"] = dataname;
    MapString["uniquename"] = dataname + to_string(data_size_millions) + "m";

#if QTRACE || PARTGRAPH
    TransEFS(dataname, MapParameter);
#endif

    if (dataname == "sift"){
        MapParameter["qsize"] = 10000;
        MapParameter["vecdim"] = 128;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Uint8";
        MapString["path_q"] = path_dataset + "query.public.10K.u8bin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.u8bin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else if (dataname == "gist"){
        if (data_size_millions > 1){
            printf("error: gist size set error.\n");
            exit(1);
        }
        MapParameter["qsize"] = 1000;
        MapParameter["vecdim"] = 960;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Float";
        MapString["path_q"] = path_dataset + "gist_query.fvecs";
        MapString["path_data"] = path_dataset + "gist_base.fvecs";
        MapString["path_gt"] = path_dataset + "gist_groundtruth.ivecs";
    } else if (dataname == "deep"){
        if (data_size_millions > 100){
            printf("error: deep size set error.\n");
            exit(1);
        }
        MapParameter["qsize"] = 10000;
        MapParameter["vecdim"] = 96;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Float";
        MapString["path_q"] = path_dataset + "query.public.10K.fbin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.fbin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else if (dataname == "turing"){
        if (data_size_millions > 100){
            printf("error: turing size set error.\n");
            exit(1);
        }
        MapParameter["qsize"] = 100000;
        MapParameter["vecdim"] = 100;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Float";
        MapString["path_q"] = path_dataset + "query100K.fbin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.fbin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else if (dataname == "spacev"){
        if (data_size_millions > 100){
            printf("error: spacev size set error.\n");
            exit(1);
        }
        MapParameter["qsize"] = 29316;
        MapParameter["vecdim"] = 100;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Int8";
        MapString["path_q"] = path_dataset + "query.i8bin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.i8bin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else{
        printf("Error, unknow dataset: %s \n", dataname.c_str()); exit(1);
    }
#if FROMBILLION
    MapString["path_data"] = "../dataset/billion/" + dataname + "/base";
#endif

    if (MapParameter["k"] > MapParameter["gt_maxnum"]){
        printf("Error, unsupport k because of bigger than gt_maxnum\n"); exit(1);
    }
}

void TransEFS(const string &dataname, map<string, size_t> &MapParameter) {
    size_t data_size_millions = MapParameter["data_size_millions"];

    // 转换baseline中同样的召回率，对应在实际配置中的结果
    if (dataname == "sift") {
        // R@10=0.95
        if (data_size_millions == 10) {
            MapParameter["efs"] = 64;
            MapParameter["interval"] = 8;
        }
    } else if (dataname == "spacev") {
        // R@10=0.90
        if (data_size_millions == 10) {
            MapParameter["efs"] = 80;
            MapParameter["interval"] = 5;
        }
    } else {
        printf("Error, unsupport dataset: %s \n", dataname.c_str()); exit(1);
    }
}