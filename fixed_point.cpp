#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <unordered_set>
#include "hnswlib/tool.h"
#include <unistd.h>

template<class T>
class FileLoading {
public:
    FileLoading(std::string file_in, bool check=true, size_t num=-1, size_t dim=-1) {
        file = new ifstream(file_in.c_str(), std::ios::binary);
        // why "file->open() is segment default" ？
        if (!file->good()) {
            printf("Error, file: %s open failed\n", file_in.c_str());
            exit(1);
        }
        int32_t num_r, dim_r;
        file->read((char *) &num_r, sizeof(int32_t));
        file->read((char *) &dim_r, sizeof(int32_t));
        if (check && !(num == num_r && dim == dim_r)) {
            printf("Error, file size: (%d, %d) no-match\n", num_r, dim_r);
            exit(1);
        }

        num_t = num_r;
        dim_v = dim_r;
        pos_head_offest = 2 * sizeof(int32_t);
        lengh_v = dim_v * sizeof(T);
    }
    ~FileLoading() {
        file->close();
    }

    void ReadVector(T* data, size_t id_beg, size_t num = 1) {
        file->seekg(id_beg * lengh_v + pos_head_offest);
        for (size_t i = 0; i < num; i++) {
            file->read((char *) (data + i * dim_v), lengh_v);
            if (file->gcount() != lengh_v) {
                printf("Read Error\n"); exit(1);
            }
        }
    }
    size_t GetNum() {
        return num_t;
    }
    size_t GetDim() {
        return dim_v;
    }

private:
    std::ifstream* file = nullptr;
    size_t num_t, dim_v, pos_head_offest;
    size_t lengh_v;
};

// get extreme value
template<typename Tdst>
float ExtremeFromFloat(std::map<string, string>& cfigStr, std::map<string, float>& cfigValue) {
    std::string file_in = cfigStr["file_in"];
    size_t vecdim = (size_t) cfigValue["vecdim"];
    size_t vecnum = (size_t) cfigValue["vecnum"];
    size_t num_train = (size_t) cfigValue["num_train"];

    FileLoading<float> File(file_in, true, vecnum, vecdim);
    float scale;

    std::unordered_set<size_t> train_list;
    srand((int)time(NULL));
    for (size_t i = 0; i < num_train; i++)
        train_list.emplace(rand() % vecnum);
    size_t num_list = train_list.size();
    printf("Train Number: %lu -> %lu\n", num_train, num_list);

    float v_max = std::numeric_limits<float>::min();
    float v_min = std::numeric_limits<float>::max();
    std::vector<float> v_list(vecdim, 0);
    std::unordered_set<size_t>::iterator iter = train_list.begin();
    for (; iter != train_list.end(); iter++) {
        File.ReadVector(v_list.data(), *iter);
        for (float value: v_list) {
            v_max = value > v_max ? value : v_max;
            v_min = value < v_min ? value : v_min;
        }
    }

    float scale_max = std::numeric_limits<Tdst>::max() / v_max;
    float scale_min = std::numeric_limits<Tdst>::min() / v_min;
    scale = std::min<float>(scale_max, scale_min);
    printf("Extreme value max: %.2f, min: %.2f\n", v_max, v_min);
    printf("Scale factor: %.1f\n", scale);
    return scale;
}

template<typename T>
void TransVector(std::vector<float>& src_v, std::vector<T>& dst_v, size_t dst_id, size_t num,
           float scale) {
    for (size_t i = 0; i < num; i++) {
        dst_v[dst_id + i] = (T) (src_v[i] * scale);
    }
}

template<typename Tdst>
void TransFileFromFloat(std::map<string, string>& cfigStr, std::map<string, float>& cfigValue) {
    // input & output file
    std::string file_in = cfigStr["file_in"];
    std::string file_out = cfigStr["file_out"];
    size_t vecdim = (size_t) cfigValue["vecdim"];
    size_t vecnum = (size_t) cfigValue["vecnum"];

    // space evaluation
    size_t mem_out_bytes = vecnum * vecdim * sizeof(Tdst);
    size_t mem_idle_bytes = 0.8 * ((size_t) cfigValue["mem_total_gib"] * (1 << 30) - mem_out_bytes);
    size_t num_batch_max = mem_idle_bytes / (vecdim * sizeof(float));
    printf("Memory[GB] available: %.0f, output: %.1f, runtime: %.1f\n",
            cfigValue["mem_total_gib"], 1.0 * mem_out_bytes / (1 << 30), 1.0 * mem_idle_bytes / (1 << 30));
    printf("Input maxinum number per batch: %lu\n", num_batch_max);

    FileLoading<float> File(file_in, true, vecnum, vecdim);
    std::vector<Tdst> Array_out(vecnum * vecdim, 0);

    // transfer to out file
    {
        float scale = cfigValue["scale"];
        printf("Transition scale factor: %.1f\n", scale);
        size_t round_t = (size_t) std::ceil(1.0 * vecnum / num_batch_max);
        printf("Transition round: %lu\n", round_t);

        for (size_t ri = 0; ri < round_t; ri++) {
            size_t id_beg = ri * num_batch_max;
            size_t num = (id_beg + num_batch_max < vecnum) ? 
                         num_batch_max : (vecnum - id_beg);
            std::vector<float> vector_batch(num * vecdim);
            File.ReadVector(vector_batch.data(), id_beg, num);
            TransVector<Tdst>(vector_batch, Array_out, id_beg, num * vecdim, scale);
            printf("[%lu / %lu]\n", (ri + 1), round_t);
        }
    }

    // output file
    WriteBinToArray<Tdst>(file_out, Array_out.data(), vecnum, vecdim);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s file_in mem_size_gib scale[0]\n", argv[0]); exit(1);
    }

    std::map<string, string> cfigStr;
    std::map<string, float> cfigValue;
    cfigStr["file_in"] = argv[1];
    cfigStr["file_out"] = cfigStr["file_in"] + "_int16";
    {
        FileLoading<float> file(cfigStr["file_in"], false);
        cfigValue["vecnum"] = file.GetNum();
        cfigValue["vecdim"] = file.GetDim();
    }
    cfigValue["num_train"] = cfigValue["vecnum"] * 0.11;
    cfigValue["mem_total_gib"] = atof(argv[2]);

    float scale = atof(argv[3]);
    if (scale == 0)
        ExtremeFromFloat<int16_t>(cfigStr, cfigValue);
    else {
        cfigValue["scale"] = scale;
        TransFileFromFloat<int16_t>(cfigStr, cfigValue);
    }

    return 0;
}