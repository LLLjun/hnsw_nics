#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>

using namespace std;

void CheckDataset(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string){

    size_t subset_size_milllions = index_parameter["subset_size_milllions"];
    string path_dataset = "dataset/" + dataname + "/";

    if (dataname == "sift"){
        index_parameter["qsize"] = 10000;
        index_parameter["vecdim"] = 128;
        index_parameter["gt_maxnum"] = 100;
        index_string["path_q"] = path_dataset + "query.public.10K.u8bin";
        index_string["path_data"] = path_dataset + dataname + to_string(subset_size_milllions) + "m/base." + to_string(subset_size_milllions) + "m.u8bin";
        index_string["path_gt"] = path_dataset + dataname + to_string(subset_size_milllions) + "m/groundtruth." + to_string(subset_size_milllions) + "m.bin";
    } else if (dataname == "gist"){
        if (subset_size_milllions > 1){
            printf("error: gist size set error.\n");
            exit(1);
        }
        index_parameter["qsize"] = 1000;
        index_parameter["vecdim"] = 960;
        index_parameter["gt_maxnum"] = 100;
        index_string["path_q"] = path_dataset + "gist_query.fvecs";
        index_string["path_data"] = path_dataset + "gist_base.fvecs";
        index_string["path_gt"] = path_dataset + "gist_groundtruth.ivecs";
    } else if (dataname == "deep"){
        if (subset_size_milllions > 100){
            printf("error: deep size set error.\n");
            exit(1);
        }
        index_parameter["qsize"] = 10000;
        index_parameter["vecdim"] = 96;
        index_parameter["gt_maxnum"] = 100;
        index_string["path_q"] = path_dataset + "query.public.10K.fbin";
        index_string["path_data"] = path_dataset + dataname + to_string(subset_size_milllions) + "m/base." + to_string(subset_size_milllions) + "m.fbin";
        index_string["path_gt"] = path_dataset + dataname + to_string(subset_size_milllions) + "m/groundtruth." + to_string(subset_size_milllions) + "m.bin";
    } else if (dataname == "turing"){
        if (subset_size_milllions > 100){
            printf("error: turing size set error.\n");
            exit(1);
        }
        index_parameter["qsize"] = 100000;
        index_parameter["vecdim"] = 100;
        index_parameter["gt_maxnum"] = 100;
        index_string["path_q"] = path_dataset + "query100K.fbin";
        index_string["path_data"] = path_dataset + dataname + to_string(subset_size_milllions) + "m/base." + to_string(subset_size_milllions) + "m.fbin";
        index_string["path_gt"] = path_dataset + dataname + to_string(subset_size_milllions) + "m/groundtruth." + to_string(subset_size_milllions) + "m.bin";
    } else{
        printf("Error, unknow dataset: %s \n", dataname.c_str());
        exit(1);
    }
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

    file_reader.read((char *) data_m, nums * dims * sizeof(data_T));
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

    file_writer.write((char *) data_m, nums * dims * sizeof(data_T));
    file_writer.close();
    printf("Write %u * %u data to %s done.\n", nums, dims, file_path.c_str());
}

template<typename data_T>
void LoadVecsToArray(std::string& file_path, data_T *data_m, uint32_t nums, uint32_t dims){
    std::ifstream file_reader(file_path.c_str(), ios::binary);
    for (size_t i = 0; i < nums; i++){
        uint32_t dims_r;
        file_reader.read((char *) &dims_r, sizeof(uint32_t));
        if (dims != dims_r){
            printf("Error, file size is error, dims_r: %u\n", dims_r);
            exit(1);
        }
        file_reader.read((char *) (data_m + i * dims), dims * sizeof(data_T));
    }
    file_reader.close();
    printf("Load %u * %u Data from %s done.\n", nums, dims, file_path.c_str());
}

template<typename data_T>
void TransIntToFloat(float *dest, data_T *src, size_t &nums, size_t &dims){
    for (size_t i = 0; i < nums; i++){
        for (size_t j = 0; j < dims; j++){
            dest[i * dims + j] = (float) src[i * dims + j];
        }
    }
}

template<typename data_T>
uint32_t compArrayCenter(const data_T *data_m, uint32_t nums, uint32_t dims){
    cout << "Comput the center point: \n";
    float *sum_m = new float[dims]();
    float *avg_m = new float[dims]();
    for (size_t i = 0; i < nums; i++){
        for (size_t j = 0; j < dims; j++){
            sum_m[j] += data_m[i * dims + j];
        }
    }
    for (size_t j = 0; j < dims; j++){
        avg_m[j] = sum_m[j] / nums;
    }

    float cur_max = std::numeric_limits<float>::max();
    uint32_t center_pt_id = 0;
// #pragma omp parallel for
    for (size_t i = 0; i < nums; i++){
        float tmp_sum = 0;
        for (size_t j = 0; j < dims; j++){
            tmp_sum += powf((data_m[i * dims + j] - avg_m[j]), 2);
        }
// #pragma omp cratical
        {
            if (tmp_sum < cur_max){
                cur_max = tmp_sum;
                center_pt_id = i;
            }
        }
    }
    cout << center_pt_id << "\n";
    return center_pt_id;
}
