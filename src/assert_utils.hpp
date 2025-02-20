#ifndef ASSERT_UTILS_HPP
#define ASSERT_UTILS_HPP

#include <cstdint>

constexpr bool hasExactNthRoot(int x, int n, int low = 1) 
{
        int high = x;
        while (low <= high)
        {
                int mid = (low + high) / 2;
                int power = 1;
                bool overflow = false;

                for (int i = 0; i < n; ++i)
                {
                        if (power > x / mid)
                        {
                                overflow = true;
                                break;
                        }
                        power *= mid;
                }
                if (!overflow && power == x) return true;
                if (overflow || power > x) high = mid - 1;
                else low = mid + 1;
        }
        return false;
}

#endif // ASSERT_UTILS_HPP
