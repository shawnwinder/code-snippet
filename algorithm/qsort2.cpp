#include <iostream>
#include <vector>
#include <utility>

using namespace std;

int partition(vector<int>& nums, int beg, int end) {
    // edge cases
    if (end - beg == 1) {
        if (nums[beg] > nums[beg+1])
            swap(nums[beg], nums[beg+1]);
        return beg;
    }

    // general cases
    int left = beg + 1;
    int right = end;
    while (left <= right) {
        if (nums[left] <= nums[beg])
            ++left;
        else if (nums[right] > nums[beg])
            --right;
        else
            swap(nums[left], nums[right]);
    }
    swap(nums[beg], nums[right]);

    return right;
}


void qsort(vector<int>& nums, int beg, int end) {
    // stop
    if (end - beg <= 0)
        return;

    // recursive
    int pos = partition(nums, beg, end);
    qsort(nums, beg, pos-1);
    qsort(nums, pos+1, end);
}


int main(int argc, char *argv[]) {
    vector<int> nums = {3,5,2,7,5,1,9,5,8,1,1,7};
    cout << "original array: ";
    for (auto n : nums)
        cout << n << ",";
    cout << endl;
    
    qsort(nums, 0, nums.size()-1);
    cout << "after sort: ";
    for (auto n : nums)
        cout << n << ",";
    cout << endl;

    return 0;
}
