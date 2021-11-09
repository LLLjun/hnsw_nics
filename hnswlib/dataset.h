#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

void CheckDataset(const string &dataname, std::map<string, size_t> &index_parameter, std::map<string, string> &index_string,
                size_t &subset_size_milllions, 
                size_t &vecsize, size_t &qsize, size_t &vecdim, size_t &gt_maxnum,
                string &path_q, string &path_data , string &path_gt){

    if (dataname == "sift"){
        qsize = 10000;
        vecdim = 128;
        gt_maxnum = 1000;
        path_q = "dataset/bigann/bigann_query.bvecs";
        path_data = "dataset/bigann/bigann_base.bvecs";
        path_gt = "dataset/bigann/gnd/idx_" + to_string(subset_size_milllions) + "M.ivecs";
    } else if (dataname == "gist"){
        if (subset_size_milllions > 1){
            printf("error: gist size set error.\n");
            exit(1);
        }
        qsize = 1000;
        vecdim = 960;
        gt_maxnum = 100;
        path_q = "dataset/gist/gist_query.fvecs";
        path_data = "dataset/gist/gist_base.fvecs";
        path_gt = "dataset/gist/gist_groundtruth.ivecs";
    } else if (dataname == "deep"){
        if (subset_size_milllions > 100){
            printf("error: deep size set error.\n");
            exit(1);
        }
        index_parameter["qsize"] = 1000;
        index_parameter["vecdim"] = 96;
        index_parameter["gt_maxnum"] = 100;
        qsize = 1000;
        vecdim = 96;
        gt_maxnum = 100;
        index_string["path_q"] = "dataset/deep/deep1B_queries.fvecs";
        index_string["path_data"] = "dataset/deep/deep_base/deep_base.fvecs";
        index_string["path_gt"] = "dataset/deep/deep_gnd/idx_" + to_string(subset_size_milllions) + "M.ivecs";      
        path_q = "dataset/deep/deep1B_queries.fvecs";
        path_data = "dataset/deep/deep_base/deep_base.fvecs";
        path_gt = "dataset/deep/deep_gnd/idx_" + to_string(subset_size_milllions) + "M.ivecs";
    } else if (dataname == "glove"){
        if (subset_size_milllions > 1){
            printf("error: glove size set error.\n");
            exit(1);
        }
        // 1193515 1193517
        vecsize = 1193515;
        qsize = 10000;
        // (25) 50 100 200
        vecdim = 25;
        gt_maxnum = 100;
        path_q = "glove/glove" + to_string(vecdim) + "d_query.fvecs";
        path_data = "glove/glove_base/glove" + to_string(vecdim) + "d_base.fvecs";
        path_gt = dataname + "/gnd/idx_" + to_string(vecdim) + "d.ivecs";
    } else{
        printf("Error, unknow dataset: %s \n", dataname.c_str());
        exit(1);
    }
    // else if (dataname == "crawl"){
    //     if (subset_size_milllions > 2){
    //         printf("error: glove size set error.\n");
    //         exit(1);
    //     }
    //     // 42 840
    //     int tokens = 42;
    //     if (tokens == 42){
    //         vecsize = 1917495;
    //     } else if(tokens == 840){
    //         vecsize = 2196018;
    //     }
    //     qsize = 10000;
    //     vecdim = 300;
    //     gt_maxnum = 100;
    //     sprintf(path_q, "crawl/crawl%dt_query.fvecs", tokens);
    //     sprintf(path_data, "crawl/crawl_base/crawl%dt_base.fvecs", tokens);
    //     sprintf(path_gt, "crawl/gnd/idx_%dt.ivecs", tokens);
    // }
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
            printf("Error, file size is error, nums_r: %u, dims_r: %u\n", nums_r, dims_r);
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
#pragma omp parallel for
    for (size_t i = 0; i < nums; i++){
        float tmp_sum = 0;
        for (size_t j = 0; j < dims; j++){
            tmp_sum += powf32((data_m[i * dims + j] - avg_m[j]), 2);
        }
#pragma omp cratical
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
