#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

using namespace std;

class MaxHeap {
    vector<int> nums;
    int heapsize;
public:
    MaxHeap(const vector<int>& arr) {
        heapsize = arr.size();
        nums.resize(heapsize + 1);
        copy(arr.begin(), arr.end(), nums.begin() + 1);
        nums[0] = 520;
        build_heap();
    }

    int size() {
        return heapsize;
    }

    void build_heap() {
        for (int i = heapsize >> 1; i >= 1; -- i) {
            sift_down(i);
        }
    }

    // the word "down" has two meaning:
    // 1. the down part of the heap is heap "ordered"
    // 2. based on 1, we need to sift the "small" one "down" to the bottom after the "swap"
    void sift_down(int p) {
        int lson = p << 1;
        int rson = p << 1 | 1;
        int largest = p;
        if (lson <= heapsize && nums[lson] > nums[largest]) {
            largest = lson;
        }
        if (rson <= heapsize && nums[rson] > nums[largest]) {
            largest = rson;
        }
        if (largest != p) {
            swap(nums[largest], nums[p]);
            sift_down(largest);
        }
    }

    int top() {
        if (heapsize <= 0) {
            cout << "empty heap ERROR!" << endl;
            return -520;
        }
        return nums[1];
    }

    void pop() {
        if (heapsize <= 0) {
            cout << "empty heap ERROR!" << endl;
        }
        /// overwrite without "swap" operation
        nums[1] = nums[heapsize];
        -- heapsize;
        nums.pop_back();  // the step can be omitted
        sift_down(1);

        /// using "swap", notice the "swap" operation can be used in "heap sort"
        // swap(nums[1], nums[heapsize]);
        // -- heapsize;
        // nums.pop_back();  // the step can be omitted
        // sift_down(1);
    }

    // the word "up" has two meaning:
    // 1. the up part of the heap is heap "ordered"
    // 2. based on 1, we need to sift the "big" one "up" to the top
    void sift_up(int son) {
        int parent = son >> 1;
        while (parent >= 1) {
            if (nums[parent] < nums[son]) {
                swap(nums[parent], nums[son]);
                son = parent;
                parent = son >> 1;
            }else {
                break;
            }
        }
    }

    void push(int val) {
        ++ heapsize;
        nums.push_back(val);
        sift_up(heapsize);
    }

    void print_heap() {
        for (int i = 0; i <= heapsize; ++ i) {
            cout << nums[i] << ",";
        }
        cout << endl;
    }
};

int main() {
    /*
     *  test sift down for heap deleting
     *  */
    // vector<int> arr = {5,3,6,9,2,4,7,1,8};
    // MaxHeap hp(arr);

    // cout << "test pop" << endl;
    // while (hp.size() > 0) {
    //     cout << hp.top() << endl;
    //     hp.pop();
    // }

    /*
     * test sift up for heap inserting
     * */
    vector<int> arr = {5,3,6,9,2,4,7,1,8};
    vector<int> tmp;
    MaxHeap hp(tmp);
    cout << "test push" << endl;
    for (auto n : arr) {
        hp.push(n);
        hp.print_heap();
    }

    return 0;
}
