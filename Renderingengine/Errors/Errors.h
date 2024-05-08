#ifndef ERRORS_H
#define ERRORS_H
#include <exception> 
#include <string> 
namespace Errors {
    class Exception : public std::exception { 
    private: 
        std::string message;
    public: 
        Exception(const char* msg);
        const char* what() const throw();
    }; 
}
#endif