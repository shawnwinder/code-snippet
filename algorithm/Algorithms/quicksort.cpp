#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <random>
#define MAXN 10000

using namespace std;

int partition(vector<int>& nums, int beg, int end) {
    if (end - beg == 2) {
        if (nums[beg] > nums[beg + 1]) {
            swap(nums[beg], nums[beg + 1]);
        }
        return beg;
    }


    default_random_engine eng;
    uniform_int_distribution<int> id(beg, end - 1);
    int pivot = id(eng); 
    swap(nums[pivot], nums[beg]);

    int left = beg + 1;
    int right = end - 1;
    while (left <= right) {
        if (nums[left] <= nums[beg]) {
            ++ left;
        }else if (nums[right] > nums[beg]) {
            -- right;
        }else {
            swap(nums[left], nums[right]);
        }
    }
    swap(nums[beg], nums[right]);

    return right;
}

void quicksort(vector<int>& nums, int beg, int end) {
    if (end - beg <= 1) {
        return;
    }

    int pos = partition(nums, beg, end);
    quicksort(nums, beg, pos);
    quicksort(nums, pos + 1, end);

}

int main() {
    // vector<int> nums = {3,6,8,4,1,9,2,8,5};
    // vector<int> nums = {0,0,1,4,1,1,-32,-54,5};
    // randomly generate 1000 length array
    default_random_engine eng;
    uniform_int_distribution<int> id(-MAXN, MAXN);
    vector<int> nums(MAXN);
    for (int i = 0; i < MAXN; ++ i) {
        nums[i] = id(eng);
    }

    quicksort(nums, 0, nums.size());

    cout << "after qsort: " << endl;
    for (auto n : nums) cout << n << ",";
    cout << endl;

    return 0;
}
