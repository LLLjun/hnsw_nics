#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <list>
#include <map>
#include <stack>
#include "profile.h"
#include "omp.h"

namespace hnswlib {
    typedef unsigned int linklistsizeint;

    template<typename dist_t, typename set_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t, set_t> *s) {

        }

        HierarchicalNSW(SpaceInterface<dist_t, set_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t, set_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
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
#if !PLATG
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
#endif
            if (linkLists_ != nullptr)
                free(linkLists_);
            if (visited_list_pool_ != nullptr)
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

#if PROEFS
            int num_iter = -1;
            int max_iter = ef_;
#endif

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

#if PROEFS
                num_iter++;
                if (num_iter >= max_iter)
                    break;
#else
                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
#endif

                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if(collect_metrics){
                    metric_hops++;
                }
#if QTRACE
                // Querytrace->addSearchPoint(getExternalLabel(current_node_id));
                Querytrace->addSearchPoint(current_node_id);
#endif
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#if QTRACE
                    // Querytrace->addNeighborBeHash(getExternalLabel(candidate_id));
                    Querytrace->addNeighborBeHash(candidate_id);
#endif
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif

                    if (!(visited_array[candidate_id] == visited_array_tag)) {
#if QTRACE
                        // Querytrace->addNeighborAfHash(getExternalLabel(candidate_id));
                        Querytrace->addNeighborAfHash(candidate_id);
#endif
                        metric_distance_computations++;

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
#if QTRACE
                Querytrace->endStep();
#endif
            }
#if QTRACE
            Querytrace->endQuery();
#endif

            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

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

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

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

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
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

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
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

        void loadIndex(const std::string &location, SpaceInterface<dist_t, set_t> *s, size_t max_elements_i=0) {


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

#if !PLATG
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
#endif

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

#if !PLATG
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
#endif
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

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
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


#if QTRACE
        QueryTrace<set_t>* Querytrace = nullptr;
#endif


        /*
            using one queue to search
        */
        struct Neighbor {
            tableint id;
            dist_t distance;
            bool flag;

            Neighbor() = default;
            Neighbor(tableint id, dist_t distance, bool f) : id{id}, distance{distance}, flag(f) {}

            inline bool operator<(const Neighbor &other) const {
                return distance < other.distance;
            }
        };

        struct Result {
            tableint id;
            dist_t distance;

            Result() = default;
            Result(tableint id, dist_t distance) : id{id}, distance{distance} {}
        };

        static inline int InsertIntoPool(Neighbor *addr, int L, Neighbor nn) {
            // find the location to insert
            int left = 0, right = L - 1;
            if (addr[left].distance > nn.distance){
                memmove((char *)&addr[left + 1], &addr[left], L * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance){
                addr[L] = nn;
                return L;
            }
            while (left < right - 1){
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)
                    right = mid;
                else
                    left = mid;
            }
            // check equal ID

            while (left > 0){
                if (addr[left].distance < nn.distance)
                    break;
                if (addr[left].id == nn.id)
                    return L + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)
                return L + 1;
            memmove((char *)&addr[right + 1], &addr[right], (L - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }


        /*
            支持rank-level的mapping
        */
#if RANKMAP
        // 不同rank内的起始搜索点
        std::vector<tableint> ept_rank;
        std::vector<vl_type> interId_to_rankLabel;
        // 仅仅只是为了转换中心点
        std::vector<std::vector<tableint>> rankId_to_interId;

        void initRankMap(){
            num_ranks = NUM_RANKS;
            // 以下的三个信息是必要的
            ept_rank.resize(num_ranks);
            interId_to_rankLabel.resize(cur_element_count);
            rankId_to_interId.resize(num_ranks);

            // 图上的点到rank，暂时采用简单的mapping方式
            size_t num_max_rank = ceil(1.0 * cur_element_count / num_ranks);
            size_t num_pad_rank = num_max_rank * num_ranks - cur_element_count;

#if MODMAP
            for (tableint in_i = 0; in_i < cur_element_count; in_i++){
                vl_type allocRankId = in_i % num_ranks;
                interId_to_rankLabel[in_i] = allocRankId;
                rankId_to_interId[allocRankId].push_back(in_i);
            }
#else
            {
                printf("only to analysis balance.\n"); exit(1);
            }
            std::vector<size_t> offest_rank_start(num_ranks);
            offest_rank_start[0] = 0;
            for (size_t i = 1; i < num_ranks; i++){
                if (i < (num_ranks - num_pad_rank + 1))
                    offest_rank_start[i] = offest_rank_start[i-1] + num_max_rank;
                else
                    offest_rank_start[i] = offest_rank_start[i-1] + (num_max_rank - 1);
            }

            for (tableint i = 0; i < cur_element_count; i++){
                for (int j = (num_ranks - 1); j >= 0; j--){
                    if (i >= offest_rank_start[j]){
                        interId_to_rankLabel[i] = j;
                        rankId_to_interId[j].push_back(i);
                        break;
                    }
                }
            }
#endif
            // 每个rank的起始点，暂时采用中心点
            size_t vecdim = *(size_t *)(dist_func_param_);
            set_t* mass_comput = new set_t[num_max_rank * vecdim]();
            for (int i = 0; i < num_ranks; i++){
                int rank_size = rankId_to_interId[i].size();
                for (int j = 0; j < rank_size; j++){
                    tableint cur_inter = rankId_to_interId[i][j];
                    memcpy(mass_comput + j * vecdim, getDataByInternalId(cur_inter), vecdim * sizeof(set_t));
                }
                int center = compArrayCenter<set_t>(mass_comput, rank_size, vecdim);
                ept_rank[i] = rankId_to_interId[i][center];
            }
            delete[] mass_comput;
            std::vector<std::vector<tableint>>().swap(rankId_to_interId);

            // RankSearch 的相关信息
            info = new InfoHardwareSearch();
            mem_rank_alloc = new tableint[num_ranks * maxM0_]();
            mem_rank_gather = new Result[num_ranks * maxM0_]();
            buffer_rank_alloc.resize(num_ranks);
            buffer_rank_gather.resize(num_ranks);
            for (int ri = 0; ri < num_ranks; ri++){
                buffer_rank_alloc[ri].first = 0;
                buffer_rank_alloc[ri].second = mem_rank_alloc + ri * maxM0_;
                buffer_rank_gather[ri].first = 0;
                buffer_rank_gather[ri].second = mem_rank_gather + ri * maxM0_;
            }
#if (OPT_SORT && OPT_VISITED)
            fetch_mem_rank_alloc = new tableint[num_ranks * maxM0_]();
            fetch_buffer_rank_alloc.resize(num_ranks);
            for (int ri = 0; ri < num_ranks; ri++){
                fetch_buffer_rank_alloc[ri].first = 0;
                fetch_buffer_rank_alloc[ri].second = fetch_mem_rank_alloc + ri * maxM0_;
            }
#endif

#if OPT_SORT
            rank_min.resize(num_ranks, std::make_pair(0, Result()));
#endif

#if STAT
            stats = new QueryStats();
            clk_query = new clk_get();
#endif
        }

        void deleteRankMap() {
            delete info;
            delete[] mem_rank_alloc;
            delete[] mem_rank_gather;
#if STAT
            delete stats;
            delete clk_query;
#endif
#if (OPT_SORT && OPT_VISITED)
            delete[] fetch_mem_rank_alloc;
#endif
        }

        int num_ranks = 1;
        QueryStats* stats = nullptr;
        clk_get* clk_query = nullptr;

        InfoHardwareSearch* info = nullptr;
        std::vector<std::pair<int, tableint*>> buffer_rank_alloc;
        std::vector<std::pair<int, Result*>> buffer_rank_gather;
        tableint* mem_rank_alloc = nullptr;
        Result* mem_rank_gather = nullptr;

#if (OPT_SORT && OPT_VISITED)
        // 由于存在地址上的冲突（在CPU模拟上），因此需要一块额外的 buffer_alloc 用于预取neighbor
        std::vector<std::pair<int, tableint*>> fetch_buffer_rank_alloc;
        tableint* fetch_mem_rank_alloc = nullptr;
#endif

#if OPT_SORT
        // 指示本轮的 search_point 的来源
        int spoint_source;
        std::vector<std::pair<int, Result>> rank_min;

        void initOptSort() {
            spoint_source = -1;
            for (int ri = 0; ri < num_ranks; ri++)
                rank_min[ri].first = 0;
        }
#endif

#if OPT_VISITED
        // 是否具备预取的条件？主要是看上一轮中queue有没有满足条件的次近点
        bool fetch_allow;
        // 本轮的 buffer_rank_alloc 是否是通过预取得到的？
        bool alloc_fetch_valid;
        //
        tableint fetch_point_last;
        tableint fetch_point;

        void initOptVisited(){
            fetch_allow = false;
            alloc_fetch_valid = false;
            fetch_point = 0;
            fetch_point_last = 0;
        }
#endif


        /*
            HNSW-plat 映射到 近存架构，主要由四部分组件。
            GetStart -> LookupVisited -> DistCalculate -> SortQueue
        */
        void InitSearch(std::vector<Neighbor>& retset, vl_type* visited_array, vl_type& visited_array_tag,
                    std::vector<std::pair<int, tableint*>>& rank_alloc, std::vector<std::pair<int, Result*>>& rank_gather) {
            // Initialize rank buffer
            for (int ri = 0; ri < num_ranks; ri++){
                rank_alloc[ri].first = 0;
                rank_gather[ri].first = 0;
            }

            // Initialize retset
#if STAT
            clk_query->reset();
#endif
            // 实际上是只计算了中心点所在的rank，其余rank空闲。
            for (int ri = 0; ri < num_ranks; ri++){
                if (ri == 0) {
                    tableint curid = ept_rank[ri];
                    dist_t curdist = fstdistfunc_(info->query_data, getDataByInternalId(curid), dist_func_param_);
                    visited_array[curid] = visited_array_tag;
                    // todo, true or false?
                    Neighbor nn(curid, curdist, true);
                    InsertIntoPool(retset.data(), info->cur_queue_size, nn);
                    info->cur_queue_size++;
                    info->p_queue_min = 0;
                }
            }
#if STAT
            stats->rank_us += clk_query->getElapsedTimeus();
            stats->n_DC_max++;
            stats->n_DC_total++;
#endif
        }

        inline tableint GetStart(std::vector<Neighbor>& retset, vl_type* visited_array, vl_type& visited_array_tag,
                            std::vector<std::pair<int, tableint*>>& rank_alloc
#if (OPT_SORT && OPT_VISITED)
                            , std::vector<std::pair<int, tableint*>>& fetch_rank_alloc
#endif
                                                                                ){
            // 获取 search_point 或者 判断终止
#if STAT
            clk_query->reset();
#endif

            tableint search_point = retset[info->p_queue_min].id;

#if OPT_SORT
            // 应当先确定 rank_min 中的最小点
            dist_t dist_rm_min = std::numeric_limits<dist_t>::max();
            int p_rm_min = -1;
            for (int ri = 0; ri < num_ranks; ri++) {
                if (rank_min[ri].first == 0)
                    continue;
                dist_t dist_rm = rank_min[ri].second.distance;
                if (dist_rm < dist_rm_min) {
                    dist_rm_min = dist_rm;
                    p_rm_min = ri;
                }
            }

            dist_t dist_ret = retset[info->p_queue_min].distance;
            if (dist_rm_min < dist_ret) {
                search_point = rank_min[p_rm_min].second.id;
                spoint_source = p_rm_min;
            } else {
                spoint_source = -1;
                // 如果 retset 未满，并且 rank_min 有意义，则需要直接填充
                if (p_rm_min != -1 && info->cur_queue_size < info->l_search) {
                    search_point = rank_min[p_rm_min].second.id;
                    spoint_source = p_rm_min;
                }
            }
#endif

            if (info->is_end){
#if OPT_SORT
                if (spoint_source != -1) {
                    info->is_end = false;
                } else {
                    info->is_done = true;
                }
#else
                info->is_done = true;
#endif
            }

            if (info->is_done) {
#if DDEBUG
                {
                    bool is_pass = true;
                    if (info->cur_queue_size != info->l_search)
                        is_pass = false;
                    if (!is_pass){
                        printf("ddebug, queue size is %d\n", info->cur_queue_size); exit(1);
                    }

                    for (int i = 0; i < info->cur_queue_size; i++){
                        if (retset[i].flag == true) {
                            is_pass = false;
                            printf("ddebug, the %d-th retset is true\n", i); exit(1);
                        }
                    }
                }
#endif
                return 0;
            }

#if OPT_VISITED
            if (alloc_fetch_valid && (search_point == fetch_point_last)) {
#if (OPT_SORT && OPT_VISITED)
                for (int ri = 0; ri < num_ranks; ri++){
                    int size = fetch_rank_alloc[ri].first;
                    rank_alloc[ri].first = size;
                    if (size == 0)
                        continue;

                    tableint* rank_point = fetch_rank_alloc[ri].second;
                    for (int ai = 0; ai < size; ai++){
                        visited_array[rank_point[ai]] = visited_array_tag;
                    }
                }
                memcpy(mem_rank_alloc, fetch_mem_rank_alloc, num_ranks * maxM0_ * sizeof(tableint));
#else
                for (int ri = 0; ri < num_ranks; ri++){
                    tableint* rank_point = rank_alloc[ri].second;
                    for (int ai = 0; ai < rank_alloc[ri].first; ai++){
                        visited_array[rank_point[ai]] = visited_array_tag;
                    }
                }
#endif
            } else {
                for (int ri = 0; ri < num_ranks; ri++)
                    rank_alloc[ri].first = 0;

                tableint *data = (tableint *) get_linklist0(search_point);
                size_t size = getListCount((linklistsizeint*) data);
                for (size_t j = 1; j <= size; j++) {
                    tableint candidate_id = *(data + j);
                    // 在硬件上可以把查表和查rankLabel合在一起
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        visited_array[candidate_id] = visited_array_tag;
                        vl_type rank_label = candidate_id % num_ranks;
                        int len = rank_alloc[rank_label].first;
                        rank_alloc[rank_label].second[len] = candidate_id;
                        rank_alloc[rank_label].first++;
                    }
                }
            }
            alloc_fetch_valid = false;
#endif


#if STAT
            stats->hlc_us += clk_query->getElapsedTimeus();
            stats->all_n_hops++;
#endif
            return search_point;
        }

        inline void LookupVisited(tableint& lookupId, vl_type* visited_array, vl_type& visited_array_tag,
                            std::vector<std::pair<int, tableint*>>& rank_alloc){
#if STAT
            clk_query->reset();
#endif

            for (int ri = 0; ri < num_ranks; ri++)
                rank_alloc[ri].first = 0;

            tableint *data = (tableint *) get_linklist0(lookupId);
            size_t size = getListCount((linklistsizeint*) data);
            for (size_t j = 1; j <= size; j++) {
                tableint candidate_id = *(data + j);
                // 在硬件上可以把查表和查rankLabel合在一起
                if (!(visited_array[candidate_id] == visited_array_tag)) {
#if (!OPT_VISITED)
                    visited_array[candidate_id] = visited_array_tag;
#endif
                    vl_type rank_label = candidate_id % num_ranks;
                    int len = rank_alloc[rank_label].first;
                    rank_alloc[rank_label].second[len] = candidate_id;
                    rank_alloc[rank_label].first++;
                }
            }
#if OPT_VISITED
            alloc_fetch_valid = true;
#endif

#if STAT
            stats->visited_us += clk_query->getElapsedTimeus();
#endif
        }

        inline void DistCalculate(std::vector<std::pair<int, tableint*>>& rank_alloc, std::vector<std::pair<int, Result*>>& rank_gather) {
#if STAT
            clk_query->reset();
#endif

            // rank-level 并行计算距离
            for (int ri = 0; ri < num_ranks; ri++){
                Result* rank_res = rank_gather[ri].second;
#if OPT_SORT
                // 内部选择最小值
                rank_min[ri].first = 0;
#endif
                for (int ai = 0; ai < rank_alloc[ri].first; ai++) {
                    tableint curid = rank_alloc[ri].second[ai];
                    dist_t curdist = fstdistfunc_(info->query_data, getDataByInternalId(curid), dist_func_param_);
                    rank_res[ai].id = curid;
                    rank_res[ai].distance = curdist;

#if OPT_SORT
                    if (rank_min[ri].first == 0) {
                        rank_min[ri].second.id = curid;
                        rank_min[ri].second.distance = curdist;
                        rank_min[ri].first = 1;
                    } else {
                        if (curdist < rank_min[ri].second.distance) {
                            rank_min[ri].second.id = curid;
                            rank_min[ri].second.distance = curdist;
                        }
                    }
#endif
                }
                rank_gather[ri].first = rank_alloc[ri].first;
                rank_alloc[ri].first = 0;
            }

#if STAT
                stats->rank_us += clk_query->getElapsedTimeus();

                int n_max = 0;
                for (std::pair<int, Result*>& brg: rank_gather){
                    n_max = std::max(n_max, brg.first);
                    stats->n_DC_total += brg.first;
                }
                stats->n_DC_max += n_max;
#endif
        }

        inline void SortQueue(std::vector<std::pair<int, Result*>>& rank_gather, std::vector<Neighbor>& retset) {
#if STAT
            clk_query->reset();
#endif

#if (!OPT_SORT)
            retset[info->p_queue_min].flag = false;
#endif
            int nk = info->cur_queue_size;
            for (int ri = 0; ri < num_ranks; ri++){
                Result* rank_res = rank_gather[ri].second;
                for (int gi = 0; gi < rank_gather[ri].first; gi++) {
                    dist_t dist = rank_res[gi].distance;
                    if ((dist >= retset[info->cur_queue_size - 1].distance) && (info->cur_queue_size == info->l_search))
                        continue;

                    tableint id = rank_res[gi].id;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), info->cur_queue_size, nn);
                    if (info->cur_queue_size < info->l_search)
                        ++info->cur_queue_size;
                    if (r < nk)
                        nk = r;
                }
            }

            // nk 位置的点不确定是否被更新过
            // 确定当前的最小点
            if (nk <= info->p_queue_min) {
#if DDEBUG
                if (!retset[nk].flag) {
                    printf("true?\n"); exit(1);
                }
#endif
                info->p_queue_min = nk;
            }
            else {
                while (info->p_queue_min < info->cur_queue_size){
                    if (retset[info->p_queue_min].flag){
                        break;
                    }
                    info->p_queue_min++;
                }
                if (info->p_queue_min == info->cur_queue_size) {
                    // 全部都是false
                    info->is_end = true;
                    info->p_queue_min = info->cur_queue_size - 1;
                }
            }

#if OPT_SORT
            // 寻找retset在第i轮的最小点
            if (!info->is_end) {
                // 将i-1轮次的最小点设为false
                retset[info->p_queue_min].flag = false;

                int submin = info->p_queue_min + 1;
                while (submin < info->cur_queue_size) {
                    if (retset[submin].flag) {
                        break;
                    }
                    submin++;
                }
                info->p_queue_min = submin;
                if (submin == info->cur_queue_size) {
                    // 对submin而言，全部都是false
                    info->is_end = true;
                    info->p_queue_min = info->cur_queue_size - 1;
                }
            }
#endif

#if OPT_VISITED
            if (fetch_allow)
                fetch_point_last = fetch_point;

            fetch_allow = false;
            if (!info->is_end) {
                int submin = info->p_queue_min + 1;
                while (submin < info->cur_queue_size) {
                    if (retset[submin].flag) {
                        fetch_allow = true;
                        fetch_point = retset[submin].id;
                        break;
                    }
                    submin++;
                }
            }
#endif

#if STAT
            stats->sort_us += clk_query->getElapsedTimeus();
#endif
        }


        /*
            input: query, k
            output: result
        */
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchParaRank(void *query_data, size_t K) {

            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            // 相关参数
            info->SetEfs(std::max(ef_, K));

            // Initialize visited list
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // Initialize priority queue
            std::vector<Neighbor> retset(info->l_search + 1);
            info->Reset(query_data);

            InitSearch(retset, visited_array, visited_array_tag, buffer_rank_alloc, buffer_rank_gather);

#if OPT_SORT
            initOptSort();
#endif
#if OPT_VISITED
            initOptVisited();
#endif

#if PROEFS
            int num_iter = 0;
            int max_iter = info->l_search;
            for (; num_iter < max_iter; num_iter++) {
#else
            while (true) {
#endif
#if (OPT_SORT && OPT_VISITED)
                tableint search_point = GetStart(retset, visited_array, visited_array_tag, buffer_rank_alloc, fetch_buffer_rank_alloc);
                if (info->is_done)
                    break;

                if (fetch_allow)
                    LookupVisited(fetch_point, visited_array, visited_array_tag, fetch_buffer_rank_alloc);

                SortQueue(buffer_rank_gather, retset);

                DistCalculate(buffer_rank_alloc, buffer_rank_gather);

#elif OPT_SORT
                tableint search_point = GetStart(retset, visited_array, visited_array_tag, buffer_rank_alloc);
                if (info->is_done)
                    break;

                SortQueue(buffer_rank_gather, retset);

                LookupVisited(search_point, visited_array, visited_array_tag, buffer_rank_alloc);

                DistCalculate(buffer_rank_alloc, buffer_rank_gather);

#elif OPT_VISITED
                tableint search_point = GetStart(retset, visited_array, visited_array_tag, buffer_rank_alloc);
                if (info->is_done)
                    break;

                DistCalculate(buffer_rank_alloc, buffer_rank_gather);

                if (fetch_allow)
                    LookupVisited(fetch_point, visited_array, visited_array_tag, buffer_rank_alloc);

                SortQueue(buffer_rank_gather, retset);

#else
                tableint search_point = GetStart(retset, visited_array, visited_array_tag, buffer_rank_alloc);
                if (info->is_done)
                    break;

                LookupVisited(search_point, visited_array, visited_array_tag, buffer_rank_alloc);

                DistCalculate(buffer_rank_alloc, buffer_rank_gather);

                SortQueue(buffer_rank_gather, retset);

#endif

#if STAT
                stats->UpdateHardwareTime();
#endif
            }

            visited_list_pool_->releaseVisitedList(vl);
            for (int i = 0; i < K; i++)
                result.push(std::pair<dist_t, labeltype>(retset[i].distance,
                            getExternalLabel(retset[i].id)));

            return result;
        };

#endif



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
                        s.insert(data[j]);
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

    };

}
