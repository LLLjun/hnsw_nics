#ifndef _CPP_KMEANS_H_
#define _CPP_KMEANS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <queue>
#include <vector>
#include "cppkmeans.h"
#include "hnswlib.h"

using namespace std;
using namespace hnswlib;

// k-means v1.0 : 优化计算代码，拓展数据量限制 

//动态创建二维数组
template<typename Gene_type>
Gene_type ** gene_array(int rowsNum, int colsNum, int dims = 1){
    Gene_type ** p = new Gene_type*[rowsNum];
	for(int i = 0; i < rowsNum; i++){
		p[i] = new Gene_type[colsNum * dims];
        if(p[i] == NULL){
            printf("Memory allocation failed\n");
            exit(1);
        }
	}
        
    return p;
}
//释放二维数组所占用的内存
template<typename Free_type>
void freearray(Free_type **p, int rowsNum){
    for(int i = 0; i < rowsNum; i++){
        delete[] p[i];
    }
    delete[] p;
}

template<typename Data_type_set_, typename Data_type_result>
class K_means{
    public:
        int K, N, D;                            //聚类的数目，数据量，数据的维数
        Data_type_set_ **data;                  //存放训练数据
        int *in_cluster;                        //标记每个点属于哪个聚类
        Data_type_set_ **cluster_center_data;   //存放每个聚类的中心点data
        int *cluster_center_id;                 //存放每个聚类的中心点id
        int *cluster_label_num;                 //存放每个聚类的数据个数
        bool is_training;
        bool is_sample_train = true;
        void *dist_func_param_;
        DISTFUNC<Data_type_result> fstdistfunc_;

        K_means(SpaceInterface<Data_type_result> *s, size_t vecdim){
            D = vecdim;
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            printf("vecdim = %lu.\n", (*(size_t*)dist_func_param_));
        }

        ~K_means(){
            delete[] in_cluster;
            freearray<Data_type_set_> (cluster_center_data, K);
            delete[] cluster_center_id;
            delete[] cluster_label_num;
        }
        // , std::allocator<uint32_t>
        float train_cluster(int k, int num, int in_num, int num_iters, std::vector<uint32_t> &data_id_list, Data_type_set_ *data_l, bool isSampleTrain = true, bool is_print=true)
        {
            is_training = true;
            is_sample_train = isSampleTrain;
            K = k; N = num;
            // 有label个优先级队列, 默认大端优先
            vector<std::priority_queue<std::pair<Data_type_result, int>>> cluster_dist_id;

            data = gene_array<Data_type_set_>(N, D);
            for (int i = 0; i < N; i++){
                for (int j = 0; j < D; j++){
                    data[i][j] = data_l[i * D + j];
                }
            }
            cluster_center_data = gene_array<Data_type_set_>(K, D);  //聚类的中心点
            cluster_center_id = new int[K]();
            in_cluster = new int[N]();
            cluster_label_num = new int[K]();

            printf("Cluster Begin: clu_label_num=%d, clu_train_num=%d, clu_vecdim=%d.\n", K, N, D);

            int clu_iter_count = 0;
            float clu_diff_bef = FLT_MAX;
            float clu_diff_now = 0.0;

            printf("cluster error:\t");
            while(fabs(clu_diff_now - clu_diff_bef) > 1e-5 && clu_iter_count < num_iters){   //比较前后两次迭代，若不相等继续迭代
                clu_diff_bef = clu_diff_now;
                // 获取当前的聚类中心点
                if (clu_iter_count == 0){
                    srand((unsigned int)(time(NULL)));  //随机初始化k个中心点
                    for(int i = 0; i < K; i++){
                        for(int j = 0; j < D; j++){
                            cluster_center_data[i][j] = data[(int)((double)N*rand()/(RAND_MAX+1.0))][j];   
                        }
                    }
                    // cluster_center_id
                    getCenter();
                } else {
                    getCenter(in_cluster);
                }
                getcluster(N, in_num, data, in_cluster, cluster_dist_id);
                
                clu_diff_now = getDifference();
                clu_iter_count++;
                // printf("The %dth difference between data and center is: %.2f\n\n", clu_iter_count, clu_diff_now);
                printf("-> %.2f\t", clu_diff_now);
                fflush(stdout);
            }
            printf("\ncluster %d iters\n\n", clu_iter_count);

            // trans sample id to origin id
            if (is_sample_train){
                for (int i = 0; i < K; i++){
                    cluster_center_id[i] = data_id_list[cluster_center_id[i]];
                }
            }

            // output result
            // if (is_print){
            //     printf("Per label number in cluster program is :\n");
            // }
            for (int cur_lab = 0; cur_lab < K; cur_lab++){
                cluster_label_num[cur_lab] = cluster_dist_id[cur_lab].size();
                if(cluster_label_num[cur_lab] == 0){
                    printf("Error: there is zero in cluster_label_num.\n");
                    exit(1);
                }
                // if (is_print){
                //     printf("%d\t", cluster_label_num[cur_lab]);
                //     if (cur_lab == (K - 1)){
                //         printf("\n");
                //     }
                // }

            }

            // free struct
            freearray<Data_type_set_>(data, N);
            (vector<std::priority_queue<std::pair<Data_type_result, int>>>()).swap(cluster_dist_id);
            if (is_print){
                printf("End of cluster train, the total iteration number of cluster is: %d.\n", clu_iter_count);  //统计迭代次数
                printf("-----------------------------\n");
            }

            return clu_diff_now;
        }


        void forward_cluster(int data_num, int cluin_limit, Data_type_set_ *search_data, int *data2label, int cluin_limit_low=1){
            if (!is_sample_train){
                printf("Error, not sample train\n");
                exit(1);
            }
            is_training = false;
            Data_type_set_ **search_array;
            search_array = gene_array<Data_type_set_>(data_num, D);
            for (int i = 0; i < data_num; i++){
                for (int j = 0; j < D; j++){
                    search_array[i][j] = search_data[i * D + j];
                }
            }
            int * cluforw_label_num = new int[K]();
            vector<std::priority_queue<std::pair<Data_type_result, int>>> cluster_dist_id;

            // printf("begin get cluster.\n");
            // 如果有需要，应当重写前向的getcluster
            getcluster(data_num, cluin_limit, search_array, data2label, cluster_dist_id, cluin_limit_low);
            // printf("end get cluster.\n");

            // output result
            for (int cur_lab = 0; cur_lab < K; cur_lab++){
                cluforw_label_num[cur_lab] = cluster_dist_id[cur_lab].size();
                if (cluforw_label_num[cur_lab] == 0){
                    printf("Error, forward cluster stage, label num is zero\n");
                    exit(1);
                }
                
                int cur_id;
                while(cluster_dist_id[cur_lab].size()){
                    cur_id = cluster_dist_id[cur_lab].top().second;
                    data2label[cur_id] = cur_lab;
                    cluster_dist_id[cur_lab].pop();
                }
            }

            printf("End of cluster inference.\n");
            printf("-----------------------------\n");

            delete[] cluforw_label_num;
            freearray<Data_type_set_>(search_array, data_num);
            (vector<std::priority_queue<std::pair<Data_type_result, int>>>()).swap(cluster_dist_id);
        }
    
    private:
        struct cmp{
            bool operator()(const pair<Data_type_result, int> p1, const pair<Data_type_result, int> p2){
                return p1.first > p2.first; //second的小值优先
            }
        };

        // 计算欧几里得距离
        Data_type_result getDistance(Data_type_set_* avector, Data_type_set_* bvector){
            return fstdistfunc_(avector, bvector, dist_func_param_, nullptr, nullptr);
        }

        // 计算每个聚类的中心点
        // update: cluster_center_id & cluster_center_data
        void getCenter(int *in_cluster){
            float **sum = gene_array<float>(K, D);
            float min_dist;
            Data_type_result **distance = gene_array<Data_type_result>(N, K);  //存放每个数据点到每个中心点的距离

            int i, j, q, count, center_id;
            for(i=0; i<K; i++)
                for(j=0; j<D; j++)
                    sum[i][j] = 0;

            for(int lab_id = 0; lab_id < K; lab_id++){
                count = 0;
                min_dist = FLT_MAX;
                
                for(int vec_id = 0; vec_id < N; vec_id++){
                    if(lab_id == in_cluster[vec_id]){
                        for(int q = 0; q < D; q++)
                            sum[lab_id][q] += data[vec_id][q];  //计算所属聚类的所有数据点的相应维数之和
                        count++;
                    }
                }
                
                for(int q = 0; q < D; q++)
                    cluster_center_data[lab_id][q] = (Data_type_set_)(sum[lab_id][q] / count);
                
                
                for(int vec_id = 0; vec_id < N; vec_id++){
                    if(lab_id == in_cluster[vec_id]){
                        distance[vec_id][lab_id] = getDistance(data[vec_id], cluster_center_data[lab_id]);
                        if(distance[vec_id][lab_id] < min_dist){
                            min_dist = distance[vec_id][lab_id];
                            center_id = vec_id;
                        }
                    }
                }

                cluster_center_id[lab_id] = center_id;
                for(int q = 0; q < D; q++)
                    cluster_center_data[lab_id][q] = data[center_id][q];

            }
            // printf("The new center of cluster is update.\n");

            freearray<float>(sum, K);
            freearray<Data_type_result>(distance, N);
        }

        void getCenter(){
            float **sum = gene_array<float>(K, D);
            float min_dist;
            Data_type_result **distance = gene_array<Data_type_result>(N, K);  //存放每个数据点到每个中心点的距离
            unordered_set<int> cent_id_list;

            int i, j, q, count, center_id;
            for(i=0; i<K; i++)
                for(j=0; j<D; j++)
                    sum[i][j] = 0;

            for(int lab_id = 0; lab_id < K; lab_id++){
                count = 0;
                min_dist = FLT_MAX;                
                
                for(int vec_id = 0; vec_id < N; vec_id++){
                    if (cent_id_list.find(vec_id) == cent_id_list.end()){
                        distance[vec_id][lab_id] = getDistance(data[vec_id], cluster_center_data[lab_id]);
                        if(distance[vec_id][lab_id] < min_dist){
                            min_dist = distance[vec_id][lab_id];
                            center_id = vec_id;
                        }
                    }
                }

                cluster_center_id[lab_id] = center_id;
                cent_id_list.insert(center_id);

                // for(int q = 0; q < D; q++)
                //     cluster_center_data[lab_id][q] = data[center_id][q];

            }
            // printf("The new center of cluster is update.\n");

            freearray<float>(sum, K);
            freearray<Data_type_result>(distance, N);
        }

        // 计算所有聚类的中心点与其数据点的距离之和
        float getDifference(){
            float sum = 0;
            for(int lab_id = 0; lab_id < K; lab_id++){
                for(int vec_id = 0; vec_id < N; vec_id++){
                    if(lab_id == in_cluster[vec_id])
                        sum += (float)getDistance(data[vec_id], cluster_center_data[lab_id]);
                }
            }
            return (sum / N);
        }

        // 把N个数据点聚类，标出每个点属于哪个聚类，主要是写data2label
        void getcluster(int data_num, int cluin_limit, Data_type_set_ **array_data, int *data2label, 
                        vector<std::priority_queue<std::pair<Data_type_result, int>>> &cluster_dist_id, int low_limit = 1){
        
            Data_type_result **distance = gene_array<Data_type_result>(data_num, K);  //存放每个数据点到每个中心点的距离

            int cur_id, rep_id, cur_label;
            
            // clear and add center
            (vector<std::priority_queue<std::pair<Data_type_result, int>>>(K)).swap(cluster_dist_id);
            for (int i = 0; i < K; i++){
                cluster_dist_id[i].push(make_pair(0, cluster_center_id[i]));
                data2label[cluster_center_id[i]] = i;
            }
            
            // 每个点距离类别的排序
            std::priority_queue<std::pair<Data_type_result, int>, vector<std::pair<Data_type_result, int>>, cmp > point_priqueue;
            std::queue<int> data_list;
            // Todo: data_num random?
            std::vector<int> random_list(data_num);
            for (int i = 0; i < data_num; i++){
                random_list[i] = i;
            }
            random_shuffle(random_list.begin(), random_list.end());
            for (int i = 0; i < data_num; i++){
                data_list.push(random_list[i]);
            }
            std::vector<int>().swap(random_list);
            // 保证不会出现空类
            bool is_empty_label = true;
            if (low_limit == 1)
                is_empty_label = false;

            size_t report_forward = data_num/10;
            bool is_report = true;

            while (data_list.size()){
                cur_id = data_list.front();
                // if (!is_training){
                //     printf("In forward get cluster, cur_id is %d\n", cur_id);
                // }
                
                data_list.pop();
                bool is_center = false;
                // 如果当前点是质心 continue
                for (int i = 0; i < K; i++){
                    if (cluster_center_id[i] == cur_id){
                        is_center = true;
                    }
                }

                if (is_center){
                    continue;
                }
                // if (!is_training){
                //     printf("In forward get cluster before comput dist\n");
                // }
                for(int j = 0; j < K; j++){
                    distance[cur_id][j] = getDistance(array_data[cur_id], cluster_center_data[j]);
                    // if (!is_training){
                    //     printf("In forward get cluster comput dist, lab_id is %d\n", j);
                    // }
                    point_priqueue.push(make_pair(distance[cur_id][j], j));
                }
                // 添加point到某个类中
                while (point_priqueue.size()){
                    // if (!is_training){
                    //     printf("In forward get cluster assign point, point_priqueue.size() is %d\n", point_priqueue.size());
                    // }
                    cur_label = point_priqueue.top().second;
                    // 优先满足下限
                    if (is_empty_label){
                        if (cluster_dist_id[cur_label].size() < low_limit){
                            cluster_dist_id[cur_label].push(make_pair(point_priqueue.top().first, cur_id));
                            data2label[cur_id] = cur_label;
                            is_report = true;

                            // handle empty flag
                            is_empty_label = false;
                            for (int j = 0; j < K; j++){
                                if (cluster_dist_id[j].size() < low_limit)
                                    is_empty_label = true;
                            }
                            break;
                        }
                    } else {
                    // {
                        if (cluster_dist_id[cur_label].size() < cluin_limit){
                            cluster_dist_id[cur_label].push(make_pair(point_priqueue.top().first, cur_id));
                            data2label[cur_id] = cur_label;
                            is_report = true;
                            break;
                        }
                        else if (point_priqueue.top().first < cluster_dist_id[cur_label].top().first){
                            if (cluster_dist_id[cur_label].size() != cluin_limit){
                                printf("cluster_dist_id is error, the size is %lu.\n", cluster_dist_id[cur_label].size());
                                exit(1);
                            }
                            data2label[cur_id] = cur_label;
                            // 为多出来的point添加回data_list中
                            rep_id = cluster_dist_id[cur_label].top().second;
                            data_list.push(rep_id);

                            cluster_dist_id[cur_label].pop();
                            cluster_dist_id[cur_label].push(make_pair(point_priqueue.top().first, cur_id));
                            is_report = false;
                            break;
                        }
                    }
                    

                    point_priqueue.pop();
                }
                // 清空剩余队列
                while (point_priqueue.size()){
                    (std::priority_queue<std::pair<Data_type_result, int>, vector<std::pair<Data_type_result, int>>, cmp>()).swap(point_priqueue);
                }
                if ((!is_training) && is_report){
                    if (data_list.size() % report_forward == 0){
                        printf("Cluster forward %d %\n", 10 * (10 - data_list.size() / report_forward));
                    }
                }
                
            }

            freearray<Data_type_result>(distance, data_num);
        }

};


#endif