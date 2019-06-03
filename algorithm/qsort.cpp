#include <iostream>
#include <vector>
#include <utility>
#include <random>

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
    swap(nums[beg], nums[pivot]);

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

void qsort(vector<int>& nums, int beg, int end) {
    if (end - beg <= 1) {
        return;
    }

    int pos = partition(nums, beg, end);
    qsort(nums, beg, pos);
    qsort(nums, pos + 1, end);
}

int main() {
    // vector<int> nums = {3,6,8,4,1,9,2,8,5};
    // vector<int> nums = {0,0,1,4,1,1,-32,-54,5};
    // randomly generate 1000 length array
    const int MAXN = 1000;
    default_random_engine eng;
    uniform_int_distribution<int> id(-MAXN, MAXN);
    vector<int> nums(MAXN);
    for (int i = 0; i < MAXN; ++ i) {
        nums[i] = id(eng);
    }
    // cout << "before qsort: " << endl;
    // for (auto n : nums) cout << n << ",";
    // cout << endl;

    qsort(nums, 0, nums.size());

    // cout << "after qsort: " << endl;
    // for (auto n : nums) cout << n << ",";
    // cout << endl;
    cout << "after qsort: " << endl;
    for (auto n : nums) cout << n << endl;

    return 0;
}
