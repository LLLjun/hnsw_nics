#ifndef _CPP_QUANTIZATION_H_
#define _CPP_QUANTIZATION_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cstring>
// #include <omp.h>

using namespace std;

/*
    0. 数据集分布分析
    1. 使用 base vectors 确定量化 scale
    2. factor_flag 存储在 vector 中，factor 应当尽可能是2的整数倍 （支持不带单粒度）
    3. 复用现有的距离计算接口
*/

template <typename TQ>
class VectorQuant{
public:
    float _scale;
    int _factor_bit, _factor_value;

    VectorQuant(size_t vecdims, bool issigned, bool uniform){
        _vecdims = vecdims;
        _uniform = uniform;
        _issigned = issigned;
        _quant_bits = 8 * sizeof(TQ);

        if (_uniform)
            printf("Quantization use [%d]-bit uniform method.\n", _quant_bits);
        else {
            _quant_bits--;
            printf("Quantization use [%d]-bit non-uniform method.\n", _quant_bits);
        }

        // todo: 目前是采用了绝对不溢出&&对称的方案
        if (_issigned) {
            _max_TQ = pow(2, _quant_bits-1) - 1;
            _min_TQ = -_max_TQ;
        } else {
            _max_TQ = pow(2, _quant_bits) - 1;
            _min_TQ = 0;
        }
    }

    ~VectorQuant() {}

    void FixInit(const float *p_data, size_t nums, size_t nums_sample);
    void FloatToFix(const float *p_data, TQ *p_data_TQ, size_t nums, bool is_eval = false);
    void GetDistrib(const float* p_data, int nums, int num_divide, vector<pair<float, float>> &v_distrib, int dim_prof = -1);

private:
    // base info
    size_t _vecdims;
    TQ _max_TQ, _min_TQ;
    float _max_real, _min_real;
    float _max_range;
    int _quant_bits;
    bool _issigned;
    //
    bool _uniform;
    float _bound;

    inline TQ round_float(float number){
        return (number > 0.0) ? floor(number + 0.5) : ceil(number - 0.5);
    }

    inline float decode_TQ(TQ data_TQ){
        if (_uniform)
            return data_TQ / _scale;
        else {
            TQ move = data_TQ & 0x0001;
            if (move == 1)
                return (data_TQ >> 1) / _scale;
            else
                return (data_TQ >> 1) / _scale / _factor_value;
        }
    }

    void GetScale(const float* p_data, int nums);
    void GetSF(const float* p_data, int nums, int num_divide = -1, float divide_val = 0);
    void EvalFix(const float *p_data, const TQ *p_data_TQ, int nums, vector<float>& quant_err);
    void GetFactor(vector<pair<float, float>>& v_distrib, vector<pair<int, float>>& v_factor);

};


    // 第一次定点，必须调用这个函数
    template <typename TQ>
    void VectorQuant<TQ>::FixInit(const float *p_data, size_t nums, size_t nums_sample){
        int num_divide = -1;
        float divide_val = 0;
        if (!_uniform){
            num_divide = 32;
            divide_val = 0.97;
        }

        if (nums > nums_sample){
            printf("Get scale: using sample data size is %d\n", nums_sample);
            float* p_data_init = new float[nums_sample * _vecdims];
            std::vector<size_t> sample_list;
            for (size_t i = 0; i < nums; i++)
                sample_list.push_back(i);
            srand((unsigned int)time(NULL));
            random_shuffle(sample_list.begin(), sample_list.end());
            sample_list.resize(nums_sample);
            std::vector<size_t>(sample_list).swap(sample_list);

            for (size_t i = 0; i < nums_sample; i++) {
                size_t data_id = sample_list[i];
                memcpy(p_data_init + i * _vecdims, p_data + data_id * _vecdims, _vecdims * sizeof(float));
            }

            GetSF(p_data_init, nums_sample, num_divide, divide_val);
            delete[] p_data_init;
        } else{
            printf("Get scale factor: using all data size is %d\n", nums);
            GetSF(p_data, nums, num_divide, divide_val);
        }
    }

    template <typename TQ>
    void VectorQuant<TQ>::FloatToFix(const float *p_data, TQ *p_data_TQ, size_t nums, bool is_eval) {
        printf("Quant %d float data to TQ is doing\n", nums);

        size_t overflow_nums = 0;

// #pragma omp parallel for
        for (size_t i = 0; i < nums; i++){
            for (size_t j = 0; j < _vecdims; j++){
                float cur_point_scale = p_data[i * _vecdims + j] * _scale;

                if (_uniform) {
                    if ((cur_point_scale >= (float)_min_TQ) && (cur_point_scale <= (float)_max_TQ))
                        p_data_TQ[i * _vecdims + j] = round_float(cur_point_scale);
                    else {
                        if (cur_point_scale > (float)_max_TQ)
                            p_data_TQ[i * _vecdims + j] = _max_TQ;
                        else if (cur_point_scale < (float)_min_TQ)
                            p_data_TQ[i * _vecdims + j] = _min_TQ;
// #pragma critical
{
                        overflow_nums++;
}
                    }
                } else {
                    TQ cur_point_T;
                    int move = 0;
                    if ((cur_point_scale >= (float)_min_TQ) && (cur_point_scale <= (float)_max_TQ)) {
                        if (abs(cur_point_scale) <= _bound)
                            cur_point_T = round_float(cur_point_scale * _factor_value);
                        else {
                            cur_point_T = round_float(cur_point_scale);
                            move = 1;
                        }
                    }
                    else {
                        if (cur_point_scale > (float)_max_TQ)
                            p_data_TQ[i * _vecdims + j] = _max_TQ;
                        else if (cur_point_scale < (float)_min_TQ)
                            p_data_TQ[i * _vecdims + j] = _min_TQ;

// #pragma critical
{
                        overflow_nums++;
}
                    }

                    // todebug
                    if (move)
                        p_data_TQ[i * _vecdims + j] = (TQ)((cur_point_T << 1) | 0x0001);
                    else
                        p_data_TQ[i * _vecdims + j] = (TQ)((cur_point_T << 1) & 0xfffe);
                }
            }
        }
        printf("Quant %d float data to TQ done, over nums is %u\n", nums, overflow_nums);

        if (is_eval){
            vector<float> quant_err;
            EvalFix(p_data, p_data_TQ, nums, quant_err);
            printf("[Fix Error] Avg: %.10f, Min: %.10f, Max: %.10f \n",
                        quant_err[0], quant_err[1], quant_err[2]);
        }
    }

    template <typename TQ>
    void VectorQuant<TQ>::GetDistrib(const float* p_data, int nums, int num_divide, vector<pair<float, float>> &v_distrib, int dim_prof) {
        v_distrib.resize(num_divide + 1, make_pair(0, 0));
        vector<uint64_t> v_sum(num_divide, 0);

        if (_issigned){
            float abs_real = max(_max_real, abs(_min_real));
            if (num_divide % 2 != 0){
                printf("Error. num_divide must be devide 2.\n");
                exit(1);
            }
            int num_d2 = num_divide / 2;
            v_distrib[num_d2].first = 0;
            for (int i = 1; i <= num_d2; i++){
                v_distrib[num_d2 + i].first = abs_real * i / num_d2;
                v_distrib[num_d2 - i].first = -v_distrib[num_d2 + i].first;
            }
        } else {
            v_distrib[0].first = 0;
            for (size_t i = 1; i <= num_divide; i++){
                v_distrib[i].first = _max_real * i / num_divide;
            }
        }

        if (dim_prof == -1) {
            for (int i = 0; i < nums; i++) {
                for (int j = 0; j < _vecdims; j++) {
                    int pos = num_divide - 1;
                    while (pos >= 0){
                        if (p_data[i * _vecdims + j] >= v_distrib[pos].first){
                            v_sum[pos]++;
                            break;
                        }
                        pos--;
                    }
                    // debug
                    if (pos < 0){
                        printf("Error, pos must >= 0 \n"); exit(1);
                    }
                }
            }

            for (int i = 0; i < num_divide; i++)
                v_distrib[i].second = (float)v_sum[i] / _vecdims;

        } else {
            if (dim_prof >= _vecdims){
                printf("Error, dim_prof must < _vecdims \n"); exit(1);
            }
            for (int i = 0; i < nums; i++) {
                int pos = num_divide - 1;
                while (pos >= 0){
                    if (p_data[i * _vecdims + dim_prof] >= v_distrib[pos].first){
                        v_distrib[pos].second++;
                        break;
                    }
                    pos--;
                }
                // debug
                if (pos < 0){
                    printf("Error, pos must >= 0 \n"); exit(1);
                }
            }
        }
        // printf distrib
        // float sum = 0;
        // for (pair<float, float> vd : v_distrib) {
        //     sum += vd.second;
        //     printf("%.3f\t%.6f\n", vd.first, (vd.second / nums));
        // }
        // printf("total: %.2f\n", sum);
        // exit(0);
    }


    template <typename TQ>
    void VectorQuant<TQ>::GetScale(const float* p_data, int nums){
        _max_real = std::numeric_limits<float>::min();
        _min_real = std::numeric_limits<float>::max();

        for (int i = 0; i < nums; i++){
            int pos_offest = i * _vecdims;
            _max_real = max(_max_real, *max_element(p_data + pos_offest, p_data + pos_offest + _vecdims));
            _min_real = min(_min_real, *min_element(p_data + pos_offest, p_data + pos_offest + _vecdims));
        }
        _max_range = max(_max_real, abs(_min_real));
        _scale = _max_TQ / _max_range;

        printf("max point(real) is %.3f, min point(real) is %.3f\n", _max_real, _min_real);
    }

    template <typename TQ>
    void VectorQuant<TQ>::GetSF(const float* p_data, int nums, int num_divide, float divide_val) {
        GetScale(p_data, nums);

        if (!_uniform) {
            vector<pair<float, float>> v_distrib;
            GetDistrib(p_data, nums, num_divide, v_distrib);

            vector<pair<int, float>> v_factor;
            GetFactor(v_distrib, v_factor);


            for (pair<int, float> v_f: v_factor){
                float radio = v_f.second / nums;
                if (radio >= divide_val){
                    _factor_bit = v_f.first;
                    _factor_value = pow(2, v_f.first);
                    break;
                }
            }
            _bound = _max_range / _factor_value;
        }

        if (_uniform){
            printf("scale: %.3f, max_range: %.3f\n", _scale, _max_range);
        } else{
            printf("scale: %.3f, max_range: %.3f, factor: %d, bound: %.3f\n",
                    _scale, _max_range, _factor_value, _bound);
        }
    }

    template <typename TQ>
    void VectorQuant<TQ>::EvalFix(const float *p_data, const TQ *p_data_TQ, int nums, vector<float>& quant_err){
        // avg, min, max
        quant_err.resize(3);
        quant_err[1] = numeric_limits<float>::max();
        quant_err[2] = numeric_limits<float>::min();

        float sumerr = 0;
        for (size_t i = 0; i < nums; i++){
            for(size_t j = 0; j < _vecdims; j++){
                float tmperr = pow((p_data[i*_vecdims+j] - decode_TQ(p_data_TQ[i*_vecdims+j])), 2);
                sumerr += tmperr;
                quant_err[1] = min(quant_err[1], tmperr);
                quant_err[2] = max(quant_err[2], tmperr);
            }
        }
        quant_err[0] = sqrt(sumerr / nums / _vecdims);
        quant_err[1] = sqrt(quant_err[1]);
        quant_err[2] = sqrt(quant_err[2]);
    }

    /*
        using non-uniform method
    */
    // v_factor.first: 移动的位数
    template <typename TQ>
    void VectorQuant<TQ>::GetFactor(vector<pair<float, float>>& v_distrib, vector<pair<int, float>>& v_factor) {
        int n_interval = v_distrib.size() - 1;
        if (_issigned)
            n_interval /= 2;

        int v_order = log2(n_interval);
        if ((int)pow(2, v_order) != n_interval){
            printf("v_distrib size must be 2^n.\n"); exit(1);
        }

        printf("order, nums\n");
        v_factor.resize(v_order + 1);
        float tmp_sum = 0;
        int order_i = 0;
        for (int i = 0; i < n_interval; i++){
            if (_issigned)
                tmp_sum += (v_distrib[n_interval+i].second + v_distrib[n_interval-1-i].second);
            else
                tmp_sum += v_distrib[i].second;

            if ((i + 1) == (int)pow(2, order_i)){
                v_factor[order_i].first = v_order - order_i;
                v_factor[order_i].second = tmp_sum;
                printf("%d, %.3f\n", v_factor[order_i].first, v_factor[order_i].second);
                order_i++;
            }
        }
    }

#endif