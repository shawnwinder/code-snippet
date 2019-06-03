#include <vector>
#include <deque>
#include <iostream>

using namespace std;

// mono non increasing queue
struct mono_queue {
    deque<int> q;
    
    void push(const vector<int>& nums, int idx) {
        if (q.empty()) {
            q.push_back(idx);
        }else {
            while (!q.empty() && nums[idx] > nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(idx);
        }
    }
    
    int front() {
        return q.front();
    }
    
    void pop() {
        q.pop_front();
    }
};

class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size) {
        vector<int> res;
        
        // edge cases
        if (num.empty()) {
            return res;
        }
        
        if (size == 0) {
            return res;
        }
        
        if (size == 1) {
            return num;
        }
        
        // general cases
        // initialize
        mono_queue q;
        for (int i = 0; i < size; ++ i) {
            q.push(num, i);
        }
        res.push_back(num[q.front()]);

        // sliding
        for (int i = 1; i <= num.size() - size; ++ i) {
            if (i > q.front()) {
                q.pop();
            }
            q.push(num, i + size - 1);
            res.push_back(num[q.front()]);
        }
        
        return res;
    }
};


int main() {
    vector<int> nums = {2,3,4,2,6,2,5,1};
    int window = 3;

    Solution sol;
    auto v = sol.maxInWindows(nums, window);

    for (auto n : v) cout << n << ",";
    cout << endl;

    return 0;
}
