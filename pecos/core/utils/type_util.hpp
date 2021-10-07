#include <cstdint>
#include <string>
#include <type_traits>

namespace pecos {

namespace type_util {

    namespace details {
        // same as __PRETTY_FUNCTION__
        template<typename T>
        static std::string pretty_name() {
            std::string str;
#if defined(__clang__)
            auto prefix   = std::string{"T = "};
            auto suffix   = std::string{","};
            auto function = std::string{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
            auto prefix   = std::string{"T = "};
            auto suffix   = std::string{";"};
            auto function = std::string{__PRETTY_FUNCTION__};
#else
# error Unsupported compiler
#endif
            auto start = function.find(prefix) + prefix.size();
            auto end = function.find(suffix, start);
            str = function.substr(start, (end - start));
            return str;
        }

        // return type_name without any template info
        template<typename T>
        static std::string name() {
            auto str = pretty_name<T>();
            auto pos = str.find("<");
            if(pos == std::string::npos) {
                return str;
            } else {
                return str.substr(0, pos);
            }
        }
        template<typename T>
        struct FullName {
            static std::string str() {
                if(std::is_same<T, void>::value) {
                    return "void";
                } else if(std::is_same<T, bool>::value) {
                    return "bool";
                } else if(std::is_same<T, char>::value) {
                    return "char";
                } else if(std::is_same<T, char16_t>::value) {
                    return "char16_t";
                } else if(std::is_same<T, int8_t>::value) {
                    return "int8_t";
                } else if(std::is_same<T, int16_t>::value) {
                    return "int16_t";
                } else if(std::is_same<T, int32_t>::value) {
                    return "int32_t";
                } else if(std::is_same<T, int64_t>::value) {
                    return "int64_t";
                } else if(std::is_same<T, uint8_t>::value) {
                    return "uint8_t";
                } else if(std::is_same<T, uint16_t>::value) {
                    return "uint16_t";
                } else if(std::is_same<T, uint32_t>::value) {
                    return "uint32_t";
                } else if(std::is_same<T, uint64_t>::value) {
                    return "uint64_t";
                } else if(std::is_same<T, float>::value) {
                    return "float";
                } else if(std::is_same<T, double>::value) {
                    return "double";
                } else if(std::is_same<T, long double>::value) {
                    return "long double";
                } else if(std::is_class<T>::value) {
                    return name<T>();
                } else {
                    return "wrong"; // should not reach to here
                }
            }
        };

        template<typename T>
        struct FullName<T&> { static std::string str() { return FullName<T>::str() + "&"; } };

        template<typename T>
        struct FullName<T&&> { static std::string str() { return FullName<T>::str() + "&&"; } };

        template<typename T>
        struct FullName<T*> { static std::string str() { return FullName<T>::str() + "*"; } };

        template<typename T>
        struct FullName<const T> { static std::string str() { return "const " + FullName<T>::str(); } };
        template<template<class> class C, class T1>
        struct FullName<C<T1>> {
            static std::string str() {
                typedef C<T1> cont_t;
                std::string ret;
                ret += name<cont_t>();
                ret += "<";
                ret += FullName<T1>::str();
                ret += ">";
                return ret;
            }
        };

        template<template<typename, typename> class C, typename T1, typename T2>
        struct FullName<C<T1, T2>> {
            static std::string str() {
                typedef C<T1, T2> cont_t;
                std::string ret;
                ret += name<cont_t>();
                ret += "<";
                ret += FullName<T1>::str();
                ret += ", ";
                ret += FullName<T2>::str();
                ret += ">";
                return ret;
            }
        };

        template<template<class, class, class> class C, class T1, class T2, class T3>
        struct FullName<C<T1, T2, T3>> {
            static std::string str() {
                typedef C<T1, T2, T3> cont_t;
                std::string ret;
                ret += name<cont_t>();
                ret += "<";
                ret += FullName<T1>::str();
                ret += ", ";
                ret += FullName<T2>::str();
                ret += ", ";
                ret += FullName<T3>::str();
                ret += ">";
                return ret;
            }
        };

       template<template<class, class, class, class> class C, class T1, class T2, class T3, class T4>
        struct FullName<C<T1, T2, T3, T4>> {
            static std::string str() {
                typedef C<T1, T2, T3, T4> cont_t;
                std::string ret;
                ret += name<cont_t>();
                ret += "<";
                ret += FullName<T1>::str();
                ret += ", ";
                ret += FullName<T2>::str();
                ret += ", ";
                ret += FullName<T3>::str();
                ret += ", ";
                ret += FullName<T4>::str();
                ret += ">";
                return ret;
            }
        };
    } // end of namespace details

    // return portabal type_name with proper template info
    template<typename T>
    static std::string full_name() {
        return details::FullName<T>::str();
    }

} // end namespace type_util
} // end namespace pecos
