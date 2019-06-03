#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    vector<int> nums(10);

    nums[2] = 2;
    for (int i = 0; i < 10; ++ i) {
        cout << nums[i] << endl;
        // cout << nums.size() << endl;
        // cout << nums.capacity() << endl;
    }

    return 0;
}
