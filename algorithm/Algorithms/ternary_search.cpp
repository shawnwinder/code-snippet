#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/*
 * usually used in finding maximun or minimun of unimodal function
 * by doing iterations when satisifying the precision requirment
*/
int ternary_search(const vector<int>& nums, int x) {
    int left = 0;
    int right = nums.size() - 1;

    while(left < right) {
        // notice the mid1 & mid2 here
        // cannot use left = (2*left + right) / 3 or right = (left + 2*right) / 3 for INTERGERS!
        // these are for float numbers!
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;
        if (nums[mid1] == x) {
            return mid1;
        }
        if (nums[mid2] == x) {
            return mid2;
        }
        if (x < nums[mid1]) {
            right = mid1 - 1;
        }else if (x > nums[mid2]) {
            left = mid2 + 1;
        }else {
            left = mid1 + 1;
            right = mid2 - 1;
        }
    }

    return -1;
}

int main() {
    vector<int> nums = {2, 3, 5, 6, -1, 20, 8, 9, 12, 13, 14};
    sort(nums.begin(), nums.end());

    int x = -1;
    cout << ternary_search(nums, x) << endl;

    return 0;
}
