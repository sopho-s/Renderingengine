#include "Errors.h"

namespace Errors {
    Exception::Exception(const char* msg) : message(msg) {}
    const char* Exception::what() const throw() {
        return message.c_str(); 
    }
}