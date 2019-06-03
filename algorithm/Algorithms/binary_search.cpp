#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// find the lower bound of value
int binary_search(const vector<int>& nums, int val) {
    // edge cases
    if (nums.empty()) {
        return -1;
    }

    // general cases
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (nums[mid] < val) {
            left = mid + 1;
        }else {
            right = mid - 1;
        }
    }

    return nums[right + 1] == val ? right + 1 : -1;
}

int main() {
    // vector<int> nums = {3,5,8,2,6,0,6,4,7,8,3,45,21,6,768,23,43,7,523,64};
    // vector<int> nums = {1,1,1,1,2};
    vector<int> nums = {1,2,2,2,2,2};
    sort(nums.begin(), nums.end());

    cout << "sorted nums: ";
    for (auto n : nums) cout << n << ",";
    cout << endl;

    // int val = 7;
    // int val = -1;
    int val = 2;
    int ret = binary_search(nums, val);
    if (ret == -1) {
        cout << "not found" << endl;
    }else {
        cout << "the lower bound of " << val << " is: " << ret << endl;
    }

    return 0;
}
