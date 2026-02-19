#pragma once

#if defined(_WIN32)
   #if defined(MIFROST_BUILD_SHARED)
      #if defined(MIFROST_EXPORTS)
         #define MIFROST_API __declspec(dllexport)
      #else
         #define MIFROST_API __declspec(dllimport)
      #endif
   #else
      #define MIFROST_API
   #endif
   #define MIFROST_LOCAL
#else
   #define MIFROST_API __attribute__((visibility("default")))
   #define MIFROST_LOCAL __attribute__((visibility("hidden")))
#endif
