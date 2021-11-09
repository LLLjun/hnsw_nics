#ifndef _CPP_QUANTIZATION_H_
#define _CPP_QUANTIZATION_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cstring>

using namespace std;

template <typename T>
class DirectQuant{
public:
    // 直接量化
    size_t _vecdims;
    float _max_real, _min_real;
    float _max_fac_coarse, _min_fac_coarse;
    float _scale_factor_coarse;
    
    // 双粒度量化
    bool _isTwoRangeQuant;
    float _max_fac_fine, _min_fac_fine;
    float _scale_factor_fine;
    T _proportion_single;       // _scale_factor_fine / _scale_factor_coarse

    size_t _vecsize, _querysize;
    size_t _coarse_table_len;
    // std::vector<std::vector<uint8_t>> CoarseTableBase;
    // std::vector<std::vector<uint8_t>> CoarseTableQuery;
    uint8_t* CoarseTableBase = nullptr;
    uint8_t* CoarseTableQuery = nullptr;
    uint8_t one_bit_to_value[8] = {128, 64, 32, 16, 8, 4, 2, 1};

    bool _isEvalFix;
    float _quant_err = 0;
    size_t _quant_nums = 0;
    size_t _overflow_nums = 0;

    bool _isQmethodPad = true;
    

    T max_T = std::numeric_limits<T>::max();
    T min_T = std::numeric_limits<T>::min();

    int _quant_bits;

    DirectQuant(size_t vecdims, bool two_range, bool qmethod_pad, bool evalfix, size_t n_base=0, size_t n_query=0){
        _vecdims = vecdims;
        _isTwoRangeQuant = two_range;
        _isQmethodPad = qmethod_pad;
        _isEvalFix = evalfix;
        _quant_bits = 8 * sizeof(T);
        
        printf("[%d]-bit Quantization.\n", _quant_bits);
        if (two_range)
            CreateCoarseTable(n_base, n_query);
    }



    // 必须是完整的一个矩阵，并添加量化粒度的标志
    void AddFullDataToFix(const float *floatp, T *fixp, size_t nums, uint8_t masstype){
        if (!_isTwoRangeQuant){
            printf("Error, Need TwoRange Quantization Table\n");
            exit(1);
        }
        // masstype: 0-basedata, 1-query
        if (masstype == 0){
            float2fix_impl(floatp, fixp, nums, CoarseTableBase);
        } else if (masstype == 1){
            float2fix_impl(floatp, fixp, nums, CoarseTableQuery);
        } else {
            printf("error\n");
            exit(1);
        }
        printf("Add full data to fix for %d mass ok\n", masstype);
    }

    void AddFullDataToFix(const float *floatp, T *fixp, size_t nums){
        if (_isTwoRangeQuant){
            printf("Error, Non TwoRange Quantization Table\n");
            exit(1);
        }
        float2fix_impl(floatp, fixp, nums);
        printf("Add full data to fix data done.\n");
    }

    uint8_t TableIDToFlag(uint8_t *CoarseTable, size_t cur_id, size_t cur_pos){
        uint8_t flag_id = (uint8_t) (cur_pos / 8);
        uint8_t bit_id = (uint8_t) (cur_pos % 8);
        uint8_t flag_add = one_bit_to_value[bit_id];

        uint8_t final_flag = CoarseTable[cur_id * _coarse_table_len + flag_id] & flag_add;

        if (final_flag == 0){
            return (uint8_t) 0;
        } else if(final_flag == flag_add){
            return (uint8_t) 1;
        } else {
            printf("error in TableIDToFlag\n");
            printf("before value: %d, after value: %d\n", 
                    CoarseTable[cur_id * _coarse_table_len + flag_id], final_flag);
            exit(1);
        }
    }

    // 已有粗粒度表格，返回缩放因子，用于计算误差
    void TableIdToFactorList(size_t table_id, float* id_factor_list,
                            uint8_t *CoarseTable){

        uint8_t table_flag;
        for (size_t i = 0; i < _vecdims; i++){
            table_flag = TableIDToFlag(CoarseTable, table_id, i);
            
            if (table_flag == 0){
                id_factor_list[i] = _scale_factor_fine;
            } else if (table_flag == 1){
                id_factor_list[i] = _scale_factor_coarse;
            } else{
                printf("error in TableIdToFactorList\n");
                printf("table flag: %d\n", table_flag);
                exit(1);
            }
        }
    }

    // 已有粗粒度表格，返回比例倍数，用于计算
    void TableIdToProportionList(size_t table_id, T* id_proportion_list, uint8_t *CoarseTable){
        // if (table_id > CoarseTable.size()){
        //     printf("error in Table Id ToProportionList\n");
        //     printf("query id: %d, CoarseTable size: %d,\n", table_id, CoarseTable.size());
        //     exit(1);
        // }

        for (size_t cur_pos = 0; cur_pos < _vecdims; cur_pos++){
            uint8_t flag_id = (uint8_t) (cur_pos / 8);
            uint8_t bit_id = (uint8_t) (cur_pos % 8);
            uint8_t flag_add = one_bit_to_value[bit_id];

            uint8_t final_flag = CoarseTable[table_id * _coarse_table_len + flag_id] & flag_add;

            if (final_flag == 0){
                id_proportion_list[cur_pos] = 1;
            } else if(final_flag == flag_add){
                id_proportion_list[cur_pos] = _proportion_single;
            } else {
                printf("error in Table Id ToProportionList\n");
                printf("table flag: %d\n", final_flag);
                exit(1);
            }
        }
    }

    void TableIdToTransFloatData(size_t table_id, float* in_pt, float* trans_pt, uint8_t *CoarseTable){
        // if (table_id > CoarseTable.size()){
        //     printf("error in Table Id ToProportionList\n");
        //     printf("query id: %d, CoarseTable size: %d,\n", table_id, CoarseTable.size());
        //     exit(1);
        // }

        for (size_t cur_pos = 0; cur_pos < _vecdims; cur_pos++){
            uint8_t flag_id = (uint8_t) (cur_pos / 8);
            uint8_t bit_id = (uint8_t) (cur_pos % 8);
            uint8_t flag_add = one_bit_to_value[bit_id];

            uint8_t final_flag = CoarseTable[table_id * _coarse_table_len + flag_id] & flag_add;

            if (final_flag == 0){
                trans_pt[cur_pos] = in_pt[cur_pos];
            } else if(final_flag == flag_add){
                trans_pt[cur_pos] = in_pt[cur_pos] * _proportion_single;
            } else {
                printf("error in Table Id ToProportionList\n");
                printf("table flag: %d\n", final_flag);
                exit(1);
            }
        }
    }

    // 第一次定点，必须调用这个函数
    void FixDataPoint(const float *floatp, size_t nums, size_t sample_nums){
        if (nums > sample_nums){
            printf("Get scale factor: using sample data size is %d\n", sample_nums);
            float* sample_datas = new float[sample_nums * _vecdims];
            std::vector<size_t> sample_list;
            for (size_t i = 0; i < nums; i++)
                sample_list.push_back(i);
            random_shuffle(sample_list.begin(), sample_list.end());
            sample_list.resize(sample_nums);
            std::vector<size_t>(sample_list).swap(sample_list);

            for (size_t i = 0; i < sample_nums; i++) {
                size_t data_id = sample_list[i];
                memcpy(sample_datas + i * _vecdims, floatp + data_id * _vecdims, _vecdims * sizeof(float));
            }
            GetFactor(sample_datas, sample_nums);
            delete[] sample_datas;
        } else{
            printf("Get scale factor: using all data size is %d\n", nums);
            GetFactor(floatp, nums);
        }
    }


    // 前提：需要获取最值
    float DatasetDistrib(const float *floatp, size_t nums, size_t num_distrib, size_t sample_nums, float divide_val){
        std::vector<std::vector<size_t>> dims_to_distrib(_vecdims);
        std::vector<float> offest(num_distrib + 1);
        printf("offest data:\n");
        if (_min_fac_coarse >= 0){
            offest[0] = 0;
            for (size_t i = 1; i <= num_distrib; i++){
                offest[i] = _max_fac_coarse * i / num_distrib;
            } 
        } else{
            if (num_distrib % 2 != 0){
                printf("Error. num_distrib must be devide 2.\n");
                exit(1);
            }
            size_t num_d2 = num_distrib / 2;
            offest[num_d2] = 0;
            for (size_t i = 1; i <= num_d2; i++){
                offest[num_d2 + i] = _max_fac_coarse * i / num_d2;
                offest[num_d2 - i] = -offest[num_d2 + i];
            }
        }
        for (size_t i = 0; i <= num_distrib; i++){
            printf("%.3f\t", offest[i]);
        } 
        printf("\n");

        // dims_to_distrib
        for (size_t i = 0; i < _vecdims; i++){
            dims_to_distrib[i].resize(num_distrib, 0);

            for (size_t j = 0; j < nums; j++){
                float dataf = floatp[i + j * _vecdims];
                size_t pos_i = 0;
                if (dataf != _min_real){
                    while(dataf > offest[pos_i]){
                        if (dataf <= offest[pos_i+1])
                            break;
                        pos_i++;
                    }
                }
                dims_to_distrib[i][pos_i]++;
            }
            {
                size_t num_t = 0;
                for (size_t j = 0; j < num_distrib; j++){
                    num_t += dims_to_distrib[i][j];
                }
                if (num_t != nums){
                    printf("dims_to_distrib sum error, dim: %d, num_t: %d, nums: %d\n", i, num_t, nums);
                    exit(1);
                }
            }
        }

        size_t tmp_num = 0;
        size_t fine_quant_nums = (size_t) nums * _vecdims * divide_val;
        float fine_quant_value;

        if (_min_fac_coarse >= 0){
            for (size_t i = 0; i < num_distrib; i++){
                for (size_t j = 0; j < _vecdims; j++){
                    tmp_num += dims_to_distrib[j][i];
                }
                if (tmp_num >= fine_quant_nums){
                    fine_quant_value = offest[i+1];
                    break;
                }
            }
        } else {
            size_t num_d2 = num_distrib / 2;
            for (size_t i = 0; i < num_d2; i++){
                for (size_t j = 0; j < _vecdims; j++){
                    tmp_num += dims_to_distrib[j][num_d2 + i];
                    if (i != 0)
                        tmp_num += dims_to_distrib[j][num_d2 - i];
                }
                if (tmp_num >= fine_quant_nums){
                    fine_quant_value = offest[num_d2+i+1];
                    break;
                }
                if (i == (num_d2 - 1))
                    fine_quant_value = offest[num_distrib];
            }
        }
        printf("fine quantization val: %.2f, expect num: %d, fine num: %d, fine value: %.3f\n",
                divide_val, fine_quant_nums, tmp_num, fine_quant_value);
        // exit(0);
        return fine_quant_value;
    }
private:
    void GetFactor(const float *floatp, size_t nums){
        _max_real = std::numeric_limits<float>::min();
        _min_real = std::numeric_limits<float>::max();

        for (size_t i = 0; i < nums; i++){
            size_t pos_offest = i * _vecdims;
            _max_real = max(_max_real, *max_element(floatp + pos_offest, floatp + pos_offest + _vecdims));
            _min_real = min(_min_real, *min_element(floatp + pos_offest, floatp + pos_offest + _vecdims));
        }
        printf("max point(real) is %.3f, min point(real) is %.3f\n", _max_real, _min_real);

        // Simple method
        _max_fac_coarse = _max_real;
        _min_fac_coarse = _min_real;

        if (_isTwoRangeQuant){
            // liujun. double factor quantization
            // todo: set _proportion_single to int, else error is very high
            _max_fac_fine = DatasetDistrib(floatp, nums, 40, nums, 0.95);
            if (_min_fac_coarse >= 0)
                _min_fac_fine = _min_real;
            else
                _min_fac_fine = -_max_fac_fine;
            printf("max point(fine scale) is %.3f, min point(fine scale) is %.3f\n", _max_fac_fine, _min_fac_fine);
        } 
        // else {
        //     // liujun. simple fix quantization
        //     _max_fac_fine = _max_fac_coarse;
        //     _min_fac_fine = _min_fac_coarse;
        // }

        printf("max point(carose scale) is %.3f, min point(carose scale) is %.3f\n", _max_fac_coarse, _min_fac_coarse);
        
        if (_isQmethodPad){
            float max_factor = (float) max_T / _max_fac_coarse;
            float min_factor = std::numeric_limits<float>::max();
            if (min_T != 0 && _min_fac_coarse != 0)
                min_factor = (float) min_T / _min_fac_coarse;
            _scale_factor_coarse = min(max_factor, min_factor);

            if (_isTwoRangeQuant){
                max_factor = (float) max_T / _max_fac_fine;
                min_factor = std::numeric_limits<float>::max();
                if (min_T != 0 && _min_fac_fine != 0)
                    min_factor = (float) min_T / _min_fac_fine;
                _scale_factor_fine = min(max_factor, min_factor);
                // to limit _scale_factor_fine
                _scale_factor_fine = (unsigned)(_scale_factor_fine / _scale_factor_coarse) * _scale_factor_coarse;
            }
        } else{ 
            // todo
            printf("Error, unsupport non padding\n");
            exit(1);
            int fixpointpos = (int) floorf32(log2f32(max_T / _max_fac_coarse));
            int fixpointneg = std::numeric_limits<int>::min();
            if (min_T != 0 && _min_fac_coarse != 0)
                fixpointneg = (int) floorf32(log2f32(min_T / _min_fac_coarse));
            int fixpoint = max(fixpointpos, fixpointneg);
            _scale_factor_coarse = pow(2, fixpoint);

            fixpointpos = (int) floorf32(log2f32(max_T / _max_fac_fine));
            fixpointneg = std::numeric_limits<int>::min();
            if (min_T != 0 && _min_fac_fine != 0)
                fixpointneg = (int) floorf32(log2f32(min_T / _min_fac_fine));
            fixpoint = max(fixpointpos, fixpointneg);
            _scale_factor_fine = pow(2, fixpoint);
        }

        if (_isTwoRangeQuant){
            // youwenti 0.185 7
            _proportion_single = (T) (_scale_factor_fine / _scale_factor_coarse);
            
            printf("max T: %d, min T: %d, carose scale factor: %.3f, fine scale factor: %.3f\n", 
                    max_T, min_T, _scale_factor_coarse, _scale_factor_fine);
            printf("_proportion_single: %d\n", _proportion_single);
        } else{
            printf("max T: %d, min T: %d, scale factor: %.3f\n", 
                    max_T, min_T, _scale_factor_coarse);
        }
    }

    void fix2floatCoarse(const T *fixp, float *floatp, size_t nums, float *cfactor){
        for (size_t i = 0; i < nums; i++){
            for (size_t j = 0; j < _vecdims; j++){
                floatp[i * _vecdims + j] = (float) fixp[i * _vecdims + j] / cfactor[i * _vecdims + j];
            }
        }
    }

    void fix2float(const T *fixp, float *floatp, size_t nums){
        for (size_t i = 0; i < nums; i++){
            for (size_t j = 0; j < _vecdims; j++){
                floatp[i * _vecdims + j] = (float) fixp[i * _vecdims + j] / _scale_factor_coarse;
            }
        }
    }

    void EvalFixCoarse(const float *floatp, const T *fixp, size_t nums,
                        uint8_t *CoarseTable){
        printf("begin eval coarse %d nums float -> fix\n", nums);

        float tmperr1 = 0;
        float tmperr2 = 0;
        float *trans_floatp = new float[nums * _vecdims];
        float *coarse_factor = new float[nums * _vecdims];
        for (size_t i = 0; i < nums; i++){
            TableIdToFactorList(i, coarse_factor + i * _vecdims, CoarseTable);
        }
        fix2floatCoarse(fixp, trans_floatp, nums, coarse_factor);

        for (size_t i = 0; i < nums; i++){
            tmperr1 = 0;
            for(size_t j = 0; j < _vecdims; j++){
                tmperr1 += powf32((floatp[i * _vecdims + j] - trans_floatp[i * _vecdims + j]), 2);
                // debug
                // printf("%.5f -> %3d -> %.5f\n", 
                //         floatp[i * _vecdims + j], fixp[i * _vecdims + j], trans_floatp[i * _vecdims + j]);
            }
            tmperr2 += powf32(tmperr1, 0.5);
        }
        
        _quant_err = (_quant_err * _quant_nums + tmperr2) / (_quant_nums + nums);
        _quant_nums += nums;

        printf("end eval coarse %d nums float -> fix\n", nums);

        delete[] trans_floatp;
        delete[] coarse_factor;
    }

    void EvalFix(const float *floatp, const T *fixp, size_t nums){
        float tmperr = 0;
        float *trans_floatp = new float[nums * _vecdims];
        fix2float(fixp, trans_floatp, nums);

        for (size_t i = 0; i < nums; i++){
            for(size_t j = 0; j < _vecdims; j++){
                tmperr += powf32((floatp[i * _vecdims + j] - trans_floatp[i * _vecdims + j]), 2);
                // debug
                // printf("%.3f -> %d -> %.3f\n", 
                //         floatp[i * _vecdims + j], fixp[i * _vecdims + j], trans_floatp[i * _vecdims + j]);
            }
            _quant_err = (_quant_err * _quant_nums + powf32(tmperr, 0.5)) / (_quant_nums + 1);
        }
        _quant_nums += nums;

        delete[] trans_floatp;
    }

    inline float fix2floatSingle(const T *fixp){
        return (float) fixp[0] / _scale_factor_coarse;
    }


    // 初始化粗粒度表格（两个）
    void CreateCoarseTable(size_t basedata_nums, size_t query_nums){
        if (!_isTwoRangeQuant){
            printf("Error, invalid use TwoRange Quantization\n");
            exit(1);
        }
        _vecsize = basedata_nums;
        _querysize = query_nums;
        _coarse_table_len = (size_t) ceil(_vecdims / 8);

        CoarseTableBase = new uint8_t[_vecsize * _coarse_table_len]();
        CoarseTableQuery = new uint8_t[_querysize * _coarse_table_len]();

        printf("Coarse Table Base size is %d, Coarse Table Query size is %d, table len is %d\n", 
                _vecsize, _querysize, _coarse_table_len);
    }

    void float2fix_impl(const float *floatp, T *fixp, size_t nums, 
                        uint8_t *CoarseTable){
        // if (CoarseTable.size() != nums){
        //     printf("float2fix_impl error, CoarseTable size: %d, nums: %d\n",
        //             CoarseTable.size(), nums);
        //     exit(1);
        // }
        printf("Fix %d nums float data to T is doing\n", nums);

        float cur_point, cur_point_T;

        // for CoarseTable
        uint8_t flag_id, bit_id, flag_add;

        for (size_t i = 0; i < nums; i++){
            for (size_t j = 0; j < _vecdims; j++){
                cur_point = floatp[i * _vecdims + j];
                flag_id = (uint8_t) (j / 8);
                bit_id = (uint8_t) (j % 8);
                flag_add = 0;
                
                if((cur_point <= _max_fac_fine) && (cur_point >= _min_fac_fine)){
                    cur_point_T = cur_point * _scale_factor_fine;
                } else{
                    cur_point_T = cur_point * _scale_factor_coarse;
                    // add coarse flag
                    flag_add = one_bit_to_value[bit_id];
                    CoarseTable[i * _coarse_table_len + flag_id] |= flag_add;
                }

                if (cur_point_T > (float) max_T){
                    fixp[i * _vecdims + j] = max_T;
                    _overflow_nums++;
                }
                else if (cur_point_T < (float) min_T){
                    fixp[i * _vecdims + j] = min_T;
                    _overflow_nums++;
                }
                else
                    fixp[i * _vecdims + j] = (T) cur_point_T;
            }
        }
        printf("Fix %d nums float data to T is done\n", nums);

        if (_isEvalFix){
            EvalFixCoarse(floatp, fixp, nums, CoarseTable);
        }
    }

    void float2fix_impl(const float *floatp, T *fixp, size_t nums){
        printf("Fix %d nums float data to T is doing\n", nums);

        float cur_point, cur_point_T;

        for (size_t i = 0; i < nums; i++){
            for (size_t j = 0; j < _vecdims; j++){
                cur_point = floatp[i * _vecdims + j];
                
                if((cur_point <= _max_fac_fine) && (cur_point >= _min_fac_fine)){
                    cur_point_T = cur_point * _scale_factor_fine;
                } else{
                    cur_point_T = cur_point * _scale_factor_coarse;
                }

                if (cur_point_T > (float) max_T){
                    fixp[i * _vecdims + j] = max_T;
                    _overflow_nums++;
                }
                else if (cur_point_T < (float) min_T){
                    fixp[i * _vecdims + j] = min_T;
                    _overflow_nums++;
                }
                else
                    fixp[i * _vecdims + j] = (T) cur_point_T;             
            }
        }
        printf("Fix %d nums float data to T is done\n", nums);

        if (_isEvalFix){
            EvalFix(floatp, fixp, nums);
        }
    }


};

#endif