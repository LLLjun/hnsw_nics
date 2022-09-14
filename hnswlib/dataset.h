#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "config.h"

using namespace std;

void CheckDataset(const string &dataname, map<string, size_t> &MapParameter, map<string, string> &MapString){

    size_t data_size_millions = MapParameter["data_size_millions"];
    string path_dataset = "../dataset/" + dataname + "/";

    if (dataname == "sift"){
        MapParameter["qsize"] = 10000;
        MapParameter["vecdim"] = 128;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Uint8";
        MapString["path_q"] = path_dataset + "query.public.10K.u8bin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.u8bin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin_brute";
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
        MapParameter["qsize"] = 10000;
        MapParameter["vecdim"] = 96;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Float";
        MapString["path_q"] = path_dataset + "query.public.10K.fbin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.fbin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else if (dataname == "turing"){
        MapParameter["qsize"] = 100000;
        MapParameter["vecdim"] = 100;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Float";
        MapString["path_q"] = path_dataset + "query100K.fbin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.fbin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else if (dataname == "spacev"){
        MapParameter["qsize"] = 29316;
        MapParameter["vecdim"] = 100;
        MapParameter["gt_maxnum"] = 100;
        MapString["format"] = "Int8";
        MapString["path_q"] = path_dataset + "query.i8bin";
        MapString["path_data"] = path_dataset + dataname + to_string(data_size_millions) + "m/base." + to_string(data_size_millions) + "m.i8bin";
        MapString["path_gt"] = path_dataset + dataname + to_string(data_size_millions) + "m/groundtruth." + to_string(data_size_millions) + "m.bin";
    } else{
        printf("Error, unknow dataset: %s \n", dataname.c_str());
        exit(1);
    }
#if FROMBILLION
    MapParameter["gt_maxnum"] = 10;
    MapString["path_data"] = "../dataset/billion/" + dataname + "/base";
    MapString["path_q"] = "../dataset/billion/" + dataname + "/query";
    MapString["path_gt"] = "../dataset/billion/" + dataname + "/groundtruth.bin";
#endif
}


// load file. store format: (uint32_t)num, (uint32_t)dim, (data_T)num * dim.
template<typename data_T>
void LoadBinToArray(std::string& file_path, data_T *data_m, uint32_t nums, uint32_t dims, bool non_header = false){
    std::ifstream file_reader(file_path.c_str(), ios::binary);
    if (!non_header){
        uint32_t nums_r, dims_r;
        file_reader.read((char *) &nums_r, sizeof(uint32_t));
        file_reader.read((char *) &dims_r, sizeof(uint32_t));
        if ((nums != nums_r) || (dims != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", file_path.c_str(), nums_r, dims_r);
            exit(1);
        }
    }

    uint32_t readsize = dims * sizeof(data_T);
    for (size_t ni = 0; ni < nums; ni++) {
        file_reader.read((char *) (data_m + dims * ni), readsize);
        if (file_reader.gcount() != readsize) {
            printf("Read Error\n"); exit(1);
        }
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

template<typename data_T>
void LoadBinToArrayIghead(std::string& file_path, data_T *data_m, uint32_t nums, uint32_t dims){
    std::ifstream file_reader(file_path.c_str(), ios::binary);

    uint32_t nums_r, dims_r;
    file_reader.read((char *) &nums_r, sizeof(uint32_t));
    file_reader.read((char *) &dims_r, sizeof(uint32_t));

    uint32_t readsize = dims * sizeof(data_T);
    for (size_t ni = 0; ni < nums; ni++) {
        file_reader.read((char *) (data_m + dims * ni), readsize);
        if (file_reader.gcount() != readsize) {
            printf("Read Error\n"); exit(1);
        }
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

// store file. store format: (uint32_t)num, (uint32_t)dim, (data_T)num * dim.
template<typename data_T>
void WriteBinToArray(std::string& file_path, const data_T *data_m, uint32_t nums, uint32_t dims, bool non_header = false){
    std::ofstream file_writer(file_path.c_str(), ios::binary);
    if (!non_header){
        file_writer.write((char *) &nums, sizeof(uint32_t));
        file_writer.write((char *) &dims, sizeof(uint32_t));
    }

    uint32_t writesize = dims * sizeof(data_T);
    for (size_t ni = 0; ni < nums; ni++) {
        file_writer.write((char *) (data_m + dims * ni), writesize);
        if (file_writer.fail() || file_writer.bad()) {
            printf("Write Error\n"); exit(1);
        }
    }
    file_writer.close();
    printf("Write %u * %u data to %s done.\n", nums, dims, file_path.c_str());
}

template<typename DTid, typename DTdist>
void WriteGroundTruth(std::string& file_path, const DTid *data_id, const DTdist *data_dist, uint32_t nums, uint32_t cols){
    std::ofstream file_writer(file_path.c_str(), ios::binary);
    file_writer.write((char *) &nums, sizeof(uint32_t));
    file_writer.write((char *) &cols, sizeof(uint32_t));

    uint32_t writesize = cols * sizeof(DTid);
    for (size_t ni = 0; ni < nums; ni++) {
        file_writer.write((char *) (data_id + cols * ni), writesize);
        if (file_writer.fail() || file_writer.bad()) {
            printf("Write Error\n"); exit(1);
        }
    }

    float* data_dist_float = new float[nums * cols];
    for (size_t ni = 0; ni < nums; ni++) {
        for (int j = 0; j < cols; j++)
            data_dist_float[ni * cols + j] = 1.0 * data_dist[ni * cols + j];
    }
    writesize = cols * sizeof(float);
    for (size_t ni = 0; ni < nums; ni++) {
        file_writer.write((char *) (data_dist_float + cols * ni), writesize);
        if (file_writer.fail() || file_writer.bad()) {
            printf("Write Error\n"); exit(1);
        }
    }
    file_writer.close();
    delete[] data_dist_float;
    printf("Write %u * %u data to %s done.\n", nums, cols, file_path.c_str());
}

template<typename data_T>
void LoadVecsToArray(std::string& file_path, data_T *data_m, uint32_t nums, uint32_t dims){
    std::ifstream file_reader(file_path.c_str(), ios::binary);
    for (size_t ni = 0; ni < nums; ni++){
        uint32_t dims_r;
        file_reader.read((char *) &dims_r, sizeof(uint32_t));
        if (dims != dims_r){
            printf("Error, file size is error, dims_r: %u\n", dims_r);
            exit(1);
        }
        file_reader.read((char *) (data_m + ni * dims), dims * sizeof(data_T));
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

template<typename data_T>
void TransIntToFloat(float *dest, data_T *src, size_t &nums, size_t &dims){
    for (size_t ni = 0; ni < nums; ni++){
        for (size_t j = 0; j < dims; j++){
            dest[ni * dims + j] = (float) src[ni * dims + j];
        }
    }
}

template<typename data_T>
uint32_t compArrayCenter(const data_T *data_m, uint32_t nums, uint32_t dims){
    cout << "Comput the center point: \n";
    float *sum_m = new float[dims]();
    float *avg_m = new float[dims]();
    for (size_t ni = 0; ni < nums; ni++){
        for (size_t j = 0; j < dims; j++){
            sum_m[j] += data_m[ni * dims + j];
        }
    }
    for (size_t j = 0; j < dims; j++){
        avg_m[j] = sum_m[j] / nums;
    }

    float cur_max = std::numeric_limits<float>::max();
    uint32_t center_pt_id = 0;
// #pragma omp parallel for
    for (size_t ni = 0; ni < nums; ni++){
        float tmp_sum = 0;
        for (size_t j = 0; j < dims; j++){
            tmp_sum += powf((data_m[ni * dims + j] - avg_m[j]), 2);
        }
// #pragma omp cratical
        {
            if (tmp_sum < cur_max){
                cur_max = tmp_sum;
                center_pt_id = ni;
            }
        }
    }
    cout << center_pt_id << "\n";
    return center_pt_id;
}
