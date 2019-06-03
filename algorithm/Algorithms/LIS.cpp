#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

using namespace std;

int binary_search(const vector<int>& nums, int x) {
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (nums[mid] == x) {
            return mid;
        }else if (nums[mid] < x) {
            left = mid + 1;
        }else {
            right = mid;
        }
    }

    return right;
}

int LIS(const vector<int>& nums) {
    // edge cases
    if (nums.empty() || nums.size() == 1) {
        return nums.size();
    }

    // general cases
    vector<int> asc;
    asc.push_back(nums[0]);
    for (int i = 1; i < nums.size(); ++ i) {
        if (nums[i] >= asc.back()) {
            asc.push_back(nums[i]);
        }else {
            // using lib function
            // auto it = lowder_bound(asc.begin(), asc.end(), nums[i]);
            // asc[it - asc.begin()] = nums[i];
            int pos = binary_search(asc, nums[i]);
            asc[pos] = nums[i];
        }
    }

    return asc.size();
}

int main() {
    // vector<int> nums = {2, 1, 5, 3, 6, 4, 8, 9, 7};
    // vector<int> nums = {1, 2, 5, 3, 6, 4, 7, 9, 10};
    vector<int> nums = {4,5,6,2,3,7};

    cout << "LIS of nums is: " << LIS(nums) << endl;

    return 0;
}
