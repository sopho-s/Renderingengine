#ifndef TESTER_H
#define TESTER_H
#include <vector>
#include <iostream>
#include <functional>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
namespace Tester {
    class Tester {
    private:
        std::vector<int> RepeatAmount;
        std::vector<int> TestType;
        std::vector<std::string> GroupNames;
        std::vector<std::string> TestNames;
        std::vector<int> TestGroup;
        std::vector<std::function<void()>> tests;
        int groupnum = 0;
    public:
        Tester();
        void AddTest(std::function<void()> newtest);
        void AddTest(std::function<void()> newtest, std::string name);
        void AddTimeTest(std::function<void()> newtest);
        void AddTimeTest(std::function<void()> newtest, std::string name);
        void AddAverageTimeTest(std::function<void()> newtest, int repam);
        void AddAverageTimeTest(std::function<void()> newtest, int repam, std::string name);
        void AddGroup(std::string name);
        void RunTests();
    };
    class PerformanceTester {
    private:
        std::vector<int> RepeatAmount;
        std::vector<int> TestType;
        std::vector<std::function<void()>> tests;
        std::vector<std::string> GroupNames;
        std::vector<std::string> TestNames;
        std::vector<int> TestGroup;
        int groupnum = 0;
    public:
        PerformanceTester();
        void AddTest(std::function<void()> newtest);
        void AddTest(std::function<void()> newtest, std::string name);
        void AddAverageTest(std::function<void()> newtest, int repam);
        void AddAverageTest(std::function<void()> newtest, int repam, std::string name);
        void AddGroup(std::string name);
        void RunTests();
    };
    template<typename T>
    void ASSERT_EQUAL(T value1, T value2);
    template<typename T>
    void ASSERT_NEAR_EQUAL(T value1, T value2, T error);
    template<typename T>
    void ASSERT_NOT_EQUAL(T value1, T value2);
    template<typename T>
    void ASSERT_GREATER(T value1, T value2);
    template<typename T>
    void ASSERT_LESSER(T value1, T value2);
    template<typename T>
    void ASSERT_LESSER_EQUAL(T value1, T value2);
    template<typename T>
    void ASSERT_GREATER_EQUAL(T value1, T value2);
    template<typename T>
    void ASSERT_TIMEOUT(T function, float time);
}
#endif