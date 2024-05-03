#include "Tester.h"

namespace Tester {
    Tester::Tester() {
        ;
    }
    PerformanceTester::PerformanceTester() {
        ;
    }
    void Tester::AddTest(std::function<void()> newtest) {
        tests.push_back(newtest);
        TestNames.push_back("");
        TestGroup.push_back(groupnum);
        TestType.push_back(0);
        RepeatAmount.push_back(1);
    }
    void PerformanceTester::AddTest(std::function<void()> newtest) {
        tests.push_back(newtest);
        TestNames.push_back("");
        TestGroup.push_back(groupnum);
        TestType.push_back(0);
        RepeatAmount.push_back(1);
    }
    void Tester::AddTest(std::function<void()> newtest, std::string name) {
        tests.push_back(newtest);
        TestNames.push_back(name);
        TestGroup.push_back(groupnum);
        TestType.push_back(0);
        RepeatAmount.push_back(1);
    }
    void PerformanceTester::AddTest(std::function<void()> newtest, std::string name) {
        tests.push_back(newtest);
        TestNames.push_back(name);
        TestGroup.push_back(groupnum);
        TestType.push_back(0);
        RepeatAmount.push_back(1);
    }
    void Tester::AddTimeTest(std::function<void()> newtest) {
        tests.push_back(newtest);
        TestNames.push_back("");
        TestGroup.push_back(groupnum);
        TestType.push_back(1);
        RepeatAmount.push_back(1);
    }
    void Tester::AddTimeTest(std::function<void()> newtest, std::string name) {
        tests.push_back(newtest);
        TestNames.push_back(name);
        TestGroup.push_back(groupnum);
        TestType.push_back(1);
        RepeatAmount.push_back(1);
    }
    void Tester::AddAverageTimeTest(std::function<void()> newtest, int repam) {
        tests.push_back(newtest);
        TestNames.push_back("");
        TestGroup.push_back(groupnum);
        TestType.push_back(2);
        RepeatAmount.push_back(repam);
    }
    void PerformanceTester::AddAverageTest(std::function<void()> newtest, int repam) {
        tests.push_back(newtest);
        TestNames.push_back("");
        TestGroup.push_back(groupnum);
        TestType.push_back(2);
        RepeatAmount.push_back(repam);
    }
    void Tester::AddAverageTimeTest(std::function<void()> newtest, int repam, std::string name) {
        tests.push_back(newtest);
        TestNames.push_back(name);
        TestGroup.push_back(groupnum);
        TestType.push_back(2);
        RepeatAmount.push_back(repam);
    }
    void PerformanceTester::AddAverageTest(std::function<void()> newtest, int repam, std::string name) {
        tests.push_back(newtest);
        TestNames.push_back(name);
        TestGroup.push_back(groupnum);
        TestType.push_back(2);
        RepeatAmount.push_back(repam);
    }
    void Tester::AddGroup(std::string name) {
        GroupNames.push_back(name);
        groupnum++;
    }
    void PerformanceTester::AddGroup(std::string name) {
        GroupNames.push_back(name);
        groupnum++;
    }
    void Tester::RunTests() {
        int passedtests = 0;
        std::vector<bool> results;
        std::vector<std::chrono::nanoseconds> times;
        auto start = std::chrono::high_resolution_clock::now();
        int count = 0;
        std::chrono::steady_clock::time_point teststart;
        std::chrono::steady_clock::time_point testend;
        // runs each test
        for(std::function<void()> currenttest : tests) {
            bool pass = true;
            // TODO: enum
            switch(TestType[count]) {
            case 0:
                teststart = std::chrono::high_resolution_clock::now();
                try {
                    currenttest();
                    results.push_back(true);
                    passedtests++;
                } catch (int e) {
                    results.push_back(false);
                }
                testend = std::chrono::high_resolution_clock::now();
                times.push_back(testend - teststart);
                break;
            case 1:
                teststart = std::chrono::high_resolution_clock::now();
                try {
                    currenttest();
                } catch (int e) {
                    pass = ~pass;
                }
                testend = std::chrono::high_resolution_clock::now();
                results.push_back(pass);
                passedtests++;
                times.push_back(testend - teststart);
                break;
            case 2:
                teststart = std::chrono::high_resolution_clock::now();
                try {
                    for (int i = 0; i < RepeatAmount[count]; i++) {
                        currenttest();
                    }
                } catch (int e) {
                    pass = ~pass;
                }
                testend = std::chrono::high_resolution_clock::now();
                results.push_back(pass);
                passedtests++;
                times.push_back((testend - teststart) / RepeatAmount[count]);
                break;
            }
            count++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        int testnum = 1;
        int currentgroup = 0;
        for(bool result : results) {
            // checks if a new group has been started
            if (currentgroup != TestGroup[testnum-1]) {
                currentgroup = TestGroup[testnum-1];
                std::cout << "\nTESTS FOR: " << GroupNames[currentgroup-1] << std::endl;
            }
            // adds colour
            if (result) {
                std::cout << "\033[1;32m";
            } else {
                std::cout << "\033[1;31m";
            }
            // prints different text depending on the test type
            // TODO: enum
            switch (TestType[testnum-1]) {
            case 0:
                if (result) {
                    std::cout << "TEST " << testnum;
                    if (TestNames[testnum-1] != "") {
                        std::cout << " '" << TestNames[testnum-1] << "'";
                    }
                    std::cout << ": PASSED,";
                } else {
                    std::cout << "TEST " << testnum;
                    if (TestNames[testnum-1] != "") {
                        std::cout << " '" << TestNames[testnum-1] << "'";
                    }
                    std::cout << ": FAILED,";
                }
                break;
            case 1:
                std::cout << "TEST " << testnum;
                if (TestNames[testnum-1] != "") {
                    std::cout << " '" << TestNames[testnum-1] << "'";
                }
                break;
            case 2:
                std::cout << "TEST " << testnum;
                if (TestNames[testnum-1] != "") {
                    std::cout << " '" << TestNames[testnum-1] << "'";
                }
                std::cout << " ON AVERAGE";
                break;
            }
            // prints the duration of the test in an appropriate format
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(times[testnum-1]);
            long long timecount = dur.count();
            if (timecount < 1000) {
                std::cout << " TEST TOOK: " << timecount << " NANOSECONDS\033[0m" << std::endl;
            } else if (timecount < 1000000) {
                timecount = (long long)(timecount / 1000);
                std::cout << " TEST TOOK: " << timecount << " MICROSECONDS\033[0m" << std::endl;
            } else {
                timecount = (long long)(timecount / 1000000);
                std::cout << " TEST TOOK: " << timecount << " MILLISECONDS\033[0m" << std::endl;
            }
            testnum++;
        }
        // prints the final duration of the tests in an appropriate format
        std::cout << "\n\nTESTS PASSED: " << passedtests << "\\" << testnum-1 << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long finalcount = duration.count();
        if (finalcount < 1000) {
            std::cout << "TESTS LASTED: " << finalcount << " NANOSECONDS\033[0m" << std::endl;
        } else if (finalcount < 1000000) {
            finalcount = (long long)(finalcount / 1000);
            std::cout << "TESTS LASTED: " << finalcount << " MICROSECONDS\033[0m" << std::endl;
        } else {
            finalcount = (long long)(finalcount / 1000000);
            std::cout << "TESTS LASTED: " << finalcount << " MILLISECONDS\033[0m" << std::endl;
        }
    }
    void PerformanceTester::RunTests() {
        std::vector<std::chrono::nanoseconds> times;
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point teststart;
        std::chrono::steady_clock::time_point testend;
        int count = 0;
        // perform each test and records it's time
        for(std::function<void()> currenttest : tests) {
            // TODO: enum
            switch(TestType[count]) {
            case 0:
                teststart = std::chrono::high_resolution_clock::now();
                currenttest();
                testend = std::chrono::high_resolution_clock::now();
                times.push_back(testend - teststart);
                break;
            case 2:
                std::cout << "tesst1" << std::endl;
                teststart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < RepeatAmount[count]; i++) {
                    std::cout << "tesst2" << std::endl;
                    currenttest();
                    std::cout << "tesst3" << std::endl;
                }
                testend = std::chrono::high_resolution_clock::now();
                times.push_back((testend - teststart) / RepeatAmount[count]);
                break;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        int testnum = 1;
        int currentgroup = 0;
        for(std::function<void()>  currenttest : tests) {
            // checks if a new group has been started
            if (currentgroup != TestGroup[testnum-1]) {
                currentgroup = TestGroup[testnum-1];
                std::cout << "\nTESTS FOR: " << GroupNames[currentgroup-1] << std::endl;
            }
            // colours text
            if (testnum > 1) {
                if (times[testnum-2] > times[testnum-1]) {
                    std::cout << "\033[1;32m";
                } else {
                    std::cout << "\033[1;31m";
                }
            }
            // prints different text depending on the test type
            // TODO: enum
            switch (TestType[testnum-1]) {
            case 0:
                std::cout << "TEST " << testnum;
                if (TestNames[testnum-1] != "") {
                    std::cout << " '" << TestNames[testnum-1] << "'";
                }
                break;
            case 2:
                std::cout << "TEST " << testnum;
                if (TestNames[testnum-1] != "") {
                    std::cout << " '" << TestNames[testnum-1] << "'";
                }
                std::cout << " ON AVERAGE";
                break;
            }
            // finds the speed change
            float increase = 0;
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(times[testnum-1]);
            long long timecount = dur.count();
            if (testnum != 1) {
                increase = ((float)times[testnum-2].count() / (float)dur.count());
                if (increase < 1) {
                    increase = -1/increase + 1;
                } else {
                    increase = increase - 1;
                }
            }
            // prints the test duration in an appropriate format
            if (timecount < 1000) {
                std::cout << " TEST TOOK: " << timecount << " NANOSECONDS";
            } else if (timecount < 1000000) {
                timecount = (long long)(timecount / 1000);
                std::cout << " TEST TOOK: " << timecount << " MICROSECONDS";
            } else {
                timecount = (long long)(timecount / 1000000);
                std::cout << " TEST TOOK: " << timecount << " MILLISECONDS";
            }
            // prints the speed change
            std::cout << ". THIS IS " << increase << "x FASTER\033[0m" << std::endl;
            testnum++;
        }
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long finalcount = duration.count();
        // prints the final duration in an appropriate format
        if (finalcount < 1000) {
            std::cout << "TESTS LASTED: " << finalcount << " NANOSECONDS\033[0m" << std::endl;
        } else if (finalcount < 1000000) {
            finalcount = (long long)(finalcount / 1000);
            std::cout << "TESTS LASTED: " << finalcount << " MICROSECONDS\033[0m" << std::endl;
        } else {
            finalcount = (long long)(finalcount / 1000000);
            std::cout << "TESTS LASTED: " << finalcount << " MILLISECONDS\033[0m" << std::endl;
        }
    }
    template<typename T>
    void ASSERT_EQUAL(T value1, T value2) {
        if (value1 != value2) {
            throw 0;
        }
    }
    template void ASSERT_EQUAL<int>(int value1, int value2);
    template void ASSERT_EQUAL<float>(float value1, float value2);
    template void ASSERT_EQUAL<bool>(bool value1, bool value2);
    template<typename T>
    void ASSERT_NEAR_EQUAL(T value1, T value2, T error) {
        if (value1 > value2 + error || value1 < value2 - error) {
            throw 0;
        }

    }
    template void ASSERT_NEAR_EQUAL<float>(float value1, float value2, float error);
    template<typename T>
    void ASSERT_NOT_EQUAL(T value1, T value2) {
        if (value1 == value2) {
            throw 0;
        }
    }
    template void ASSERT_NOT_EQUAL<int>(int value1, int value2);
    template void ASSERT_NOT_EQUAL<float>(float value1, float value2);
    template void ASSERT_NOT_EQUAL<bool>(bool value1, bool value2);
    template<typename T>
    void ASSERT_GREATER(T value1, T value2) {
        if (value1 <= value2) {
            throw 0;
        }
    }
    template void ASSERT_GREATER<int>(int value1, int value2);
    template void ASSERT_GREATER<float>(float value1, float value2);
    template void ASSERT_GREATER<bool>(bool value1, bool value2);
    template<typename T>
    void ASSERT_LESSER(T value1, T value2) {
        if (value1 >= value2) {
            throw 0;
        }
    }
    template void ASSERT_LESSER<int>(int value1, int value2);
    template void ASSERT_LESSER<float>(float value1, float value2);
    template void ASSERT_LESSER<bool>(bool value1, bool value2);
    template<typename T>
    void ASSERT_LESSER_EQUAL(T value1, T value2) {
        if (value1 > value2) {
            throw 0;
        }
    }
    template void ASSERT_LESSER_EQUAL<int>(int value1, int value2);
    template void ASSERT_LESSER_EQUAL<float>(float value1, float value2);
    template void ASSERT_LESSER_EQUAL<bool>(bool value1, bool value2);
    template<typename T>
    void ASSERT_GREATER_EQUAL(T value1, T value2) {
        if (value1 < value2) {
            throw 0;
        }
    }
    template void ASSERT_GREATER_EQUAL<int>(int value1, int value2);
    template void ASSERT_GREATER_EQUAL<float>(float value1, float value2);
    template void ASSERT_GREATER_EQUAL<bool>(bool value1, bool value2);
    template<typename T>
    void ASSERT_TIMEOUT(T function, std::chrono::nanoseconds time) {
        auto start = std::chrono::high_resolution_clock::now();
        function();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        if (duration.count() > time.count()) {
            throw 0;
        }
    }
    template void ASSERT_TIMEOUT<std::function<void()>>(std::function<void()> function, std::chrono::nanoseconds time);
    template void ASSERT_TIMEOUT<std::function<int()>>(std::function<int()> function, std::chrono::nanoseconds time);
    template void ASSERT_TIMEOUT<std::function<bool()>>(std::function<bool()> function, std::chrono::nanoseconds time);
    template void ASSERT_TIMEOUT<std::function<float()>>(std::function<float()> function, std::chrono::nanoseconds time);
}