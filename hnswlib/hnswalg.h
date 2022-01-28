#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include "config.h"
#include "profile.h"
#include <chrono>

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {

        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            has_deletions_=false;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

#if IOF1
            OF1.resize(max_elements_, 0);
            IF1.resize(max_elements_, 0);
#endif

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;


        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;


        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        bool has_deletions_;


        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;
        mutable std::atomic<long> metric_hops_L;

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if(collect_metrics){
                    metric_hops++;
                    // metric_distance_computations+=size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
#if NEWMETRIC
                        metric_distance_computations++;
#endif

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        unsigned dms, ncf, reach_step;
        void setHeuristic2(unsigned Dms, unsigned Ncf, unsigned Reach_step){
            dms = Dms;
            ncf = Ncf;
            reach_step = Reach_step;
        }

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M, int level = 0, tableint insert_node=0, bool isCollect = false) {
            if (top_candidates.size() < M) {
                return;
            }
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;

            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

#if (RLNG && RLDT)
            std::vector<std::vector<int>> rldt_list;
            rldt_list.resize(queue_closest.size());
            // std::unordered_set<tableint> reach_range_set;

            std::unordered_set<unsigned> connect_node;
            std::unordered_map<int, std::pair<unsigned, int>> connect_list;

            std::vector<std::pair<dist_t, tableint>> closest_list;
            while (queue_closest.size() > 0){
                closest_list.push_back(queue_closest.top());
                queue_closest.pop();
            }

            size_t vdim = *((size_t *)dist_func_param_);
            float *diffData = new float[vdim]();
            float *vec_o = (float *)getDataByInternalId(insert_node);
            for (unsigned i = 0; i < closest_list.size(); i++){
                std::pair<dist_t, tableint> curent_pair = closest_list[i];
                float *vec_n = (float *)getDataByInternalId(curent_pair.second);

                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, GreaterByFirst> set_sort;

                for (int j = 0; j < vdim; j++){
                    diffData[j] = vec_n[j] - vec_o[j];
                    if (set_sort.size() < dms)
                        set_sort.emplace(std::abs(diffData[j]), (j + 1) * signf(diffData[j]));
                    else {
                        if (std::abs(diffData[j]) > set_sort.top().first){
                            set_sort.pop();
                            set_sort.emplace(std::abs(diffData[j]), (j + 1) * signf(diffData[j]));
                        }
                    }
                }
                
                // rldt_list[i].end() is the largest
                while (!set_sort.empty()){
                    rldt_list[i].push_back(set_sort.top().second);
                    set_sort.pop();
                }
            }
            delete[] diffData;

            // first allo
            for (unsigned i = 0; i < closest_list.size(); i++){
                if (return_list.size() >= M)
                    break;
                
                std::pair<dist_t, tableint> curent_pair = closest_list[i];
                bool good = true;
                for (int key : rldt_list[i]){
                    if (connect_list.find(key) != connect_list.end()){
                        good = false;
                        break;
                    }
                }
                if (good){
                    for (int key : rldt_list[i])
                        connect_list.emplace(key, std::make_pair(return_list.size(), ncf));
                    connect_node.emplace(i);

                    return_list.push_back(curent_pair);
                    // addNeighborToSet(curent_pair.second, level, reach_range_set, reach_step);
                }
            }

            // second allo
            for (unsigned i = 0; i < closest_list.size(); i++){
                if (return_list.size() >= M)
                    break;
                if (connect_node.find(i) != connect_node.end())
                    continue;
                
                std::pair<dist_t, tableint> curent_pair = closest_list[i];
                bool good = true;
                for (int key : rldt_list[i]){
                    auto iter = connect_list.find(key);
                    if (iter != connect_list.end()){
                        tableint conflict_id = return_list[iter->second.first].second;
                        if (isNeighbor(curent_pair.second, conflict_id, level, reach_step) == 1 || (iter->second.second <= 0)){
                            good = false;
                            break;
                        }
                    }
                }
                if (good){
                    for (int key : rldt_list[i]){
                        auto iter = connect_list.find(key);
                        if (iter != connect_list.end()){
                            std::pair<unsigned, bool> c_pair = iter->second;
                            connect_list[key] = std::make_pair(c_pair.first, (c_pair.second - 1));
                        } else {
                            connect_list.emplace(key, std::make_pair(return_list.size(), ncf));
                        }
                    }
                    connect_node.emplace(i);

                    return_list.push_back(curent_pair);
                    // addNeighborToSet(curent_pair.second, level, reach_range_set, reach_step);
                }
            }

            // second allo other
            // for (unsigned i = 0; i < closest_list.size(); i++){
            //     if (return_list.size() >= M)
            //         break;
            //     if (connect_node.find(i) != connect_node.end())
            //         continue;

            //     std::pair<dist_t, tableint> curent_pair = closest_list[i];
            //     bool good = true;
            //     for (std::pair<dist_t, tableint> second_pair : return_list) {
            //         if (isNeighbor(curent_pair.second, second_pair.second, level)) {
            //             good = false;
            //             break;
            //         }
            //     }
            //     if (good) {
            //         return_list.push_back(curent_pair);
            //         connect_node.emplace(i);
            //     }
            // }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }

#elif (RLCU && RLDT)
            std::vector<std::pair<dist_t, tableint>> closest_list;
            std::vector<std::vector<unsigned>> reach_list;
            std::priority_queue<std::pair<unsigned, int>> reach_num;
            std::unordered_set<int> delete_list;

            reach_list.resize(queue_closest.size());
            while (queue_closest.size() > 0){
                closest_list.push_back(queue_closest.top());
                queue_closest.pop();
            }
            for (int i = 0; i < closest_list.size(); i++){
                tableint base_id = closest_list[i].second;
                for (int j = 0; j < closest_list.size(); j++){
                    if (i == j)
                        continue;
                    tableint compare_id = closest_list[j].second;
                    if (isNeighbor(compare_id, base_id, level)){
                        reach_list[i].push_back(j);
                    }
                }
                reach_num.emplace(reach_list[i].size(), i);
            }

            int seq_i = 0;
            while (!(return_list.size() >= M || (seq_i == closest_list.size() && reach_num.empty()))){
                for (int i = seq_i; i < closest_list.size(); i++){
                    seq_i = i + 1;
                    if (delete_list.find(i) == delete_list.end()){
                        return_list.push_back(closest_list[i]);
                        
                        delete_list.emplace(i);
                        for (unsigned j = 0; j < reach_list[i].size(); j++)
                            delete_list.emplace(reach_list[i][j]);
                        break;
                    }
                }

                while (!reach_num.empty()){
                    int i = reach_num.top().second;
                    reach_num.pop();
                    if (delete_list.find(i) == delete_list.end()){
                        return_list.push_back(closest_list[i]);
                        
                        delete_list.emplace(i);
                        for (unsigned j = 0; j < reach_list[i].size(); j++)
                            delete_list.emplace(reach_list[i][j]);
                        break;
                    }
                }
            }
            
            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }

#elif RLDT
            std::unordered_multimap<int, unsigned> bucket_base;

            size_t vdim = *((size_t *)dist_func_param_);
            float *diffData = new float[vdim]();
            float *vec_o = (float *)getDataByInternalId(insert_node);

            while (queue_closest.size()){
                if (return_list.size() >= M)
                    break;

                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                queue_closest.pop();
                float *vec_n = (float *)getDataByInternalId(curent_pair.second);

                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, GreaterByFirst> set_sort;

                for (int j = 0; j < vdim; j++){
                    diffData[j] = vec_n[j] - vec_o[j];
                    if (set_sort.size() < dms)
                        set_sort.emplace(std::abs(diffData[j]), (j + 1) * signf(diffData[j]));
                    else {
                        if (std::abs(diffData[j]) > set_sort.top().first){
                            set_sort.pop();
                            set_sort.emplace(std::abs(diffData[j]), (j + 1) * signf(diffData[j]));
                        }
                    }
                }
                
                // key_list.end() is the largest
                std::vector<int> key_list;
                while (!set_sort.empty()){
                    key_list.push_back(set_sort.top().second);
                    set_sort.pop();
                }

                bool good = true;
                for (int key: key_list){
                    auto iter_connect = bucket_base.find(key);
                    if (iter_connect != bucket_base.end()){
                        int len = bucket_base.count(key);
                        for (int i = 0; i < len; i++){
                            tableint conflict_id = return_list[iter_connect->second].second;
                            // compare neighbor connection is very important.
                            if (isNeighbor(curent_pair.second, conflict_id, level)){
                                good = false;
                                break;
                            }
                            iter_connect++;
                        }
                    }
                    if (!good)
                        break;
                }
                if (good){
                    bucket_base.emplace(key_list[dms-1], return_list.size());
                    return_list.push_back(curent_pair);
                }

            }
            delete[] diffData;

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
#else
            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
#endif
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        std::string graph_type = "base";

        bool isDggb = false;
        unsigned *dggb_in = nullptr;
        unsigned *dggb_out = nullptr;
        void setGlobalDegree(bool isGlobalDegree){
            isDggb = isGlobalDegree;
            dggb_in = new unsigned[max_elements_]();
            dggb_out = new unsigned[max_elements_]();
        }

#if IOF1
        mutable std::vector<unsigned> OF1;
        mutable std::vector<unsigned> IF1;
#endif


        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
#if EXI
            std::vector<tableint> ex_list;
            ex_list.reserve(top_candidates.size());
#endif
            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
#if EXI
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_copy;
            while (top_candidates.size() > 0) {
                top_candidates_copy.emplace(top_candidates.top());
                if (top_candidates.size() < maxM0_)
                    ex_list.push_back(top_candidates.top().second);
                top_candidates.pop();
            }
            getNeighborsByHeuristic2(top_candidates_copy, M_, level, cur_c);
            if (top_candidates_copy.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");
            while (top_candidates_copy.size() > 0) {
                selectedNeighbors.push_back(top_candidates_copy.top().second);
                top_candidates_copy.pop();
            }
#else
            getNeighborsByHeuristic2(top_candidates, M_, level, cur_c);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }
#endif

#if IOF1
            OF1[cur_c] += selectedNeighbors.size();
#endif

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }

#if EXI
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                ex_list.push_back(selectedNeighbors[idx]);
            }
            std::unordered_set<tableint> rever_node;
            unsigned IF1_cur_c = 0;
            for (size_t idx = 0; idx < ex_list.size(); idx++) {
                if (IF1_cur_c >= M_)
                    break;
                
                tableint cur_i =  ex_list[ex_list.size() - idx - 1];

                if (rever_node.find(cur_i) != rever_node.end()){
                    continue;
                }
                if (idx > M_){
                    bool no_link = true;
                    for (auto iter = rever_node.begin(); iter != rever_node.end(); iter++){
                        if (isNeighbor(*iter, cur_i, level)){
                            no_link = false;
                            break;
                        }
                    }
                    if (!no_link)
                        continue;
                }
                rever_node.emplace(cur_i);
#else
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                tableint cur_i =  selectedNeighbors[idx];
                // tableint cur_i =  selectedNeighbors[selectedNeighbors.size() - idx - 1];    
#endif

                std::unique_lock <std::mutex> lock(link_list_locks_[cur_i]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(cur_i);
                else
                    ll_other = get_linklist(cur_i, level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (cur_i == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[cur_i])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `cur_i` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if ((graph_type == "base") && (sz_link_list_other < Mcurmax)) {
                        // lower
                        // for (int jj = 0; jj < sz_link_list_other; jj++){
                        //     if (isNeighbor(cur_c, data[jj], level)){
                        //         no_connect = false;
                        //         break;
                        //     }
                        // }

                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
#if EXI
                        IF1_cur_c++;
#endif
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(cur_i),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(cur_i),
                                                 dist_func_param_), data[j]);
                        }
                        getNeighborsByHeuristic2(candidates, Mcurmax, level, cur_i);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }
                        setListCount(ll_other, indx);
#if EXI
                        if (isNeighbor(cur_c, cur_i, level))
                            IF1_cur_c++;
#endif
                    }
                }
#if IOF1
                if (isNeighbor(cur_c, cur_i, level)){
                    IF1[cur_c]++;
                }
#endif
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (has_deletions_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {


            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements=max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);


            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);




            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);


            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);


            visited_list_pool_ = new VisitedListPool(1, max_elements);


            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            has_deletions_=false;

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    has_deletions_=true;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
//        static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            has_deletions_=true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
//                    if (neigh == internalId)
//                        continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

#if PROFILE
        mutable double tb_search = 0;
        mutable double tb_sort = 0;
#endif

        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);
                    
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel;
#if PLATG
            curlevel = 0;
#else      
            curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;
#endif
            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");
#if PROFILE
                    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    tb_search += std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
                    s = std::chrono::steady_clock::now();
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                    tb_sort += std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
#else
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
#endif
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) {
#if PLATG
                printf("Error, current graph is plat\n");
                exit(1);
#endif
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops_L++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            if (has_deletions_) {
                top_candidates=searchBaseLayerST<true,true>(
                        currObj, query_data, std::max(ef_, k));
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(
                        currObj, query_data, std::max(ef_, k));
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        /*
            input: constructed graph
            output: reconstruct graph
        */
        // void reConstructionGraph(size_t resize, unsigned *relist){
        //     size_t num_reselect;
        //     tableint *node_select = nullptr;

        //     if (resize == 0){
        //         num_reselect = cur_element_count / M_;
        //         printf("reselect %u node for hierarchy graph \n", num_reselect);
        //         // only get 0 layer outdegree
        //         node_select = new tableint[num_reselect]();
        //         // std::vector<tableint> node_select(num_reselect, 0);

        //         std::vector<std::queue<tableint>> idg_list;
        //         std::vector<std::queue<tableint>> odg_list;
        //         getDegreeLayer0(idg_list, odg_list);
        //         // for (size_t i = 0; i < cur_element_count; i++){
        //         //     linklistsizeint *ll_cur = get_linklist0(i);
        //         //     int size = getListCount(ll_cur);
        //         //     odg_list[size].emplace(i);
        //         // }
        //         size_t ci = 0;
        //         size_t degree_range = odg_list.size() - 1;
        //         for (int dd = degree_range; dd > 0; dd--){
        //             while (!odg_list[dd].empty()){
        //                 node_select[ci] = odg_list[dd].front();
        //                 // node_select.push_back(odg_list[dd].front());
        //                 odg_list[dd].pop();
        //                 ci++;
        //                 if (ci == num_reselect){
        //                     printf("using min out degree: %d \n", dd);
        //                     break;
        //                 }
        //             }
        //             if (ci == num_reselect)
        //                 break;
        //         }
        //     } else {
        //         num_reselect = resize;
        //         printf("reselect %u node for hierarchy graph \n", num_reselect);
        //         node_select = new tableint[num_reselect]();
        //         memcpy(node_select, relist, num_reselect * sizeof(unsigned));
        //     }

        //     // construct hierarchy structure
        //     initHierarchy();
        //     for (size_t i = 0; i < num_reselect; i++){
        //         tableint cur_c = node_select[i];
        //         addPointHierarchy(cur_c, -1);
        //     }
        //     printf("Reconstruct hierarchy is done.\n");
        // }

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.emplace(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

        /*
            input: labeltype
            output: indegree and in node
        */
        float getDegreeDistri(std::vector<size_t> &dstb_in, std::vector<size_t> &dstb_out, bool isprint = false){
            std::vector<size_t>().swap(dstb_in);
            std::vector<size_t>().swap(dstb_out);
            unsigned *node_in = new unsigned[cur_element_count]();
            unsigned *node_out = new unsigned[cur_element_count]();

            for (size_t i = 0; i < cur_element_count; i++){
                for(size_t l = 0; l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    
                    node_out[i] += size;
                    for (int j = 0; j < size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        node_in[data[j]]++;
                    }
                }
            }

            float out_total = 0;
            for (size_t i = 0; i < cur_element_count; i++){
                out_total += node_out[i];
                if (dstb_in.size() < (node_in[i] + 1))
                    dstb_in.resize((node_in[i] + 1), 0);
                dstb_in[node_in[i]]++;
                if (dstb_out.size() < (node_out[i] + 1))
                    dstb_out.resize((node_out[i] + 1), 0);
                dstb_out[node_out[i]]++;
            }
            delete[] node_in;
            delete[] node_out;
            // printf
            if (isprint){
                printf("In Degree:\n");
                for (size_t i = 0; i < dstb_in.size(); i++)
                    printf("%u\n", dstb_in[i]);
                printf("Out Degree\n");
                for (size_t i = 0; i < dstb_out.size(); i++)
                    printf("%u\n", dstb_out[i]);
            }

            float odg_avg = (float) out_total / cur_element_count;
            printf("average outdegree: %.3f \n", odg_avg);
            return odg_avg;
        }

        /*
            input: graph
            output: indegree and outdegree relation [out * in]
        */
        unsigned ** getDegreeRelation(){
            unsigned **matrix_dg = gene_array<unsigned>(maxM0_ + 1, maxM0_ + 1);
            unsigned *node_in = new unsigned[cur_element_count]();
            unsigned *node_out = new unsigned[cur_element_count]();

            for (size_t i = 0; i < cur_element_count; i++){
                for(size_t l = 0; l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    
                    node_out[i] += size;
                    for (int j = 0; j < size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        node_in[data[j]]++;
                    }
                }
            }
            for (size_t i = 0; i < cur_element_count; i++){
                unsigned row_id = node_out[i];
                unsigned col_id = node_in[i] > maxM0_ ? maxM0_ : node_in[i];
                matrix_dg[row_id][col_id]++;
            }

            printf("relation matrix for out * in degree: \n");
            unsigned n_val = 0;
            for (size_t i = 0; i <= maxM0_; i++){
                for (size_t j = 0; j<= maxM0_; j++){
                    printf("%u\t", matrix_dg[i][j]);
                    n_val += matrix_dg[i][j];
                }
                printf("\n");
            }
            if (n_val != cur_element_count){
                printf("Error, n_val: %u \n", n_val);
                exit(1);
            }

            // freearray<unsigned>(matrix_dg, maxM0_ + 1);
            printf("gene degree relation matrix done.\n");
            return matrix_dg;
        }

        /*
            comput average neighbor distance
        */
        float compAverageNeighDist(){
            float dist_average = 0;
#pragma omp parallel for
            for (size_t it_i = 0; it_i < cur_element_count; it_i++){
                linklistsizeint *ll_cur = get_linklist0(it_i);
                int ll_size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);

                float dist_cur = 0;
                for (size_t ng_i = 0; ng_i < ll_size; ng_i++){
                    dist_cur += fstdistfunc_(getDataByInternalId(it_i), getDataByInternalId(data[ng_i]), dist_func_param_);
                }
                dist_cur /= ll_size;
#pragma omp critical
                {
                    dist_average += dist_cur;
                }
            }
            return (dist_average / cur_element_count);
        }

        /*
            input: graph
            output: specical point average neighbor distance
        */
        float compNeighborDistByDegree(){
            unsigned *node_in = new unsigned[cur_element_count]();
            unsigned *node_out = new unsigned[cur_element_count]();
            std::vector<tableint> need_comp;

            for (size_t i = 0; i < cur_element_count; i++){
                for(size_t l = 0; l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    
                    node_out[i] += size;
                    for (int j = 0; j < size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        node_in[data[j]]++;
                    }
                }
            }
            for (size_t i = 0; i < cur_element_count; i++){
                unsigned row_id = node_out[i];
                unsigned col_id = node_in[i] > maxM0_ ? maxM0_ : node_in[i];
                if (col_id < 3)
                    need_comp.push_back(i);
            }
            printf("num: %u \n", need_comp.size());
            delete[] node_in;
            delete[] node_out;

            float dist_average = 0;
#pragma omp parallel for
            for (size_t i = 0; i < need_comp.size(); i++){
                tableint it_i = need_comp[i];
                linklistsizeint *ll_cur = get_linklist0(it_i);
                int ll_size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);

                float dist_cur = 0;
                for (size_t ng_i = 0; ng_i < ll_size; ng_i++){
                    dist_cur += fstdistfunc_(getDataByInternalId(it_i), getDataByInternalId(data[ng_i]), dist_func_param_);
                }
                dist_cur /= ll_size;
#pragma omp critical
                {
                    dist_average += dist_cur;
                }
            }
            return (dist_average / need_comp.size());
        }

        int isNeighbor(const tableint xi, tableint xc, int level, unsigned step = 1, bool init = true){
            std::vector<tableint> ngh_first_step;
            
            linklistsizeint *ll_cur = get_linklist_at_level(xc, level);
            int size = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);

            for (int j = 0; j < size; j++){
                if (data[j] == xi)
                    return 1;
                ngh_first_step.push_back(data[j]);
            }

            for (tableint ngh_fst: ngh_first_step){
                linklistsizeint *ll_cur = get_linklist_at_level(ngh_fst, level);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);

                for (int j = 0; j < size; j++){
                    if (data[j] == xi)
                        return 1;
                }
            }

            return 0;
            
            // if (step == 1){
            //     linklistsizeint *ll_cur = get_linklist_at_level(xc, level);
            //     int size = getListCount(ll_cur);
            //     tableint *data = (tableint *) (ll_cur + 1);

            //     for (int j = 0; j < size; j++){
            //         if (data[j] == xi)
            //             return 1;
            //     }
            //     if (init)
            //         return 0;
            //     else
            //         return -1;
            // } else if (step > 1){
            //     step--;
            //     linklistsizeint *ll_cur = get_linklist_at_level(xc, level);
            //     int size = getListCount(ll_cur);
            //     tableint *data = (tableint *) (ll_cur + 1);

            //     for (int j = 0; j < size; j++){
            //         if (data[j] == xi)
            //             return 1;
            //     }
            //     for (int j = 0; j < size; j++){
            //         if (isNeighbor(xi, data[j], level, step, false) == 1)
            //             return 1;
            //     }
            //     if (init)
            //         return 0;
            //     else
            //         return -1;
            // } else {
            //     printf("Error\n");
            //     exit(1);
            // }
        }

        void addNeighborToSet(const tableint &xc, int &level, std::unordered_set<tableint> &set, unsigned step){
            if (step == 1){
                linklistsizeint *ll_cur = get_linklist_at_level(xc, level);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);

                for (int j = 0; j < size; j++)
                    set.emplace(data[j]);
            } else {
                if (step > 1){
                    step--;
                    linklistsizeint *ll_cur = get_linklist_at_level(xc, level);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);

                    for (int j = 0; j < size; j++){
                        set.emplace(data[j]);
                        addNeighborToSet(data[j], level, set, step);
                    }
                }
            }
        }

        int isInReachRange(const tableint &xi, std::unordered_set<tableint> &set){
            if (set.find(xi) != set.end())
                return 1;
            else
                return 0;
        }

#if (IOF1 || EXI)
        void getDegreePerPoint(std::vector<unsigned> &IDG, std::vector<unsigned> &ODG){
            std::vector<unsigned>().swap(IDG);
            std::vector<unsigned>().swap(ODG);
            IDG.resize(cur_element_count, 0);
            ODG.resize(cur_element_count, 0);

            for (size_t i = 0; i < cur_element_count; i++){
                for(size_t l = 0; l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    
                    ODG[i] += size;
                    for (int j = 0; j < size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        IDG[data[j]]++;
                    }
                }
            }
        }
#endif

        /*
            test sort and multi-add
        */
        inline int signf(float &pf){
            if (pf >= 0)
                return 1;
            else
                return -1;
        }
        struct GreaterByFirst {
            constexpr bool operator()(std::pair<float, size_t> const &a,
                                      std::pair<float, size_t> const &b) const noexcept {
                return a.first > b.first;
            }
        };
        void heap_sort_topk(std::vector<std::pair<float, size_t>> &set_unsort, size_t nums){
            size_t nums_all = set_unsort.size();
            if (nums_all < nums){
                printf("Error, the set size must larger than %u \n", nums);
                exit(1);
            }
            std::make_heap(set_unsort.begin(), set_unsort.begin() + nums);
            for (size_t i = nums; i < nums_all; i++){
                if (set_unsort[i].first > set_unsort[0].first){
                    set_unsort[0].first = set_unsort[i].first;
                    std::make_heap(set_unsort.begin(), set_unsort.begin() + nums);
                }
            }
            std::sort_heap(set_unsort.begin(), set_unsort.begin() + nums);
        }

        void testSortMultiadd(){

            size_t vsize = 1000;
            size_t vdim = *((size_t *)dist_func_param_);
            float *massA = new float[vsize * vdim]();
            float *massB = new float[vsize * vdim]();
            for (size_t i = 0; i < vsize; i++){
                for (size_t j = 0; j < vdim; j++){
                    massA[i * vdim + j] = ((float) rand() / RAND_MAX);
                    massB[i * vdim + j] = ((float) rand() / RAND_MAX);
                }
            }
            printf("Test with mass: %u * %u \n", vsize, vdim);

            std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
            // multi add
            for (size_t i = 0; i < vsize; i++){
                float *vecA = massA + i * vdim;
                float *vecB = massB + i * vdim;
                volatile float d = 0;
                // d = fstdistfunc_(vecA, vecB, dist_func_param_);
                for (size_t j = 0; j < vdim; j++){
                    d += (vecA[j] - vecB[j]) * (vecA[j] - vecB[j]);
                }
            }
            double t_mul = std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
            printf("Comput time per vector: %.3f us\n", (t_mul / vsize * 1e6));

            // sort
            s = std::chrono::steady_clock::now();
            for (size_t i = 0; i < vsize; i++){
                float *vecA = massA + i * vdim;
                float *vecB = massB + i * vdim;

                // use heap sort
                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, GreaterByFirst> set_sort;
                for (size_t j = 0; j < vdim; j++){
                    vecA[j] -= vecB[j];
                    if (set_sort.size() < 5)
                        set_sort.emplace(std::abs(vecA[j]), (j + 1) * signf(vecA[j]));
                    else {
                        if (std::abs(vecA[j]) > set_sort.top().first){
                            set_sort.pop();
                            set_sort.emplace(std::abs(vecA[j]), (j + 1) * signf(vecA[j]));
                        }
                    }
                }

                int key = 0;
                float prop = 1;
                while (!set_sort.empty()){
                    if (set_sort.size() == 1){
                        key = set_sort.top().second;
                        if (prop != 0)
                            prop = set_sort.top().first / prop;
                        else
                            prop = std::numeric_limits<float>::max();
                        break;
                    }
                    prop = set_sort.top().first;
                    set_sort.pop();
                }

                // use max for 1
                // for (size_t j = 0; j < vdim; j++){
                //     vecA[j] -= vecB[j];
                //     vecB[j] = std::abs(vecA[j]);
                // }
                // volatile int pos = std::max_element(vecB, vecB + vdim) - vecB;
                // size_t kk = pos * 10 + signf(vecA[pos]);
                // std::unordered_set<size_t> g;
                // g.emplace(kk);
            }
            double t_sort = std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
            printf("Sort time per vector: %.3f us\n", (t_sort / vsize * 1e6));
        }

    };

}
