#ifndef LLM_C_H
#define LLM_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// 导出宏：Windows下导出，其他平台兼容
#ifdef MNN_C_EXPORTS
    // 仅当编译 libmnn.dll 时定义 MNN_C_EXPORTS，此时用 dllexport
    #define MNN_C_API __declspec(dllexport)
#else
    #define MNN_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct LlmHandle {
    void* llm_ptr;
} LlmHandle;

// C API LLM_Context struct
typedef struct LLM_Context {
    // Basic info
    int prompt_len;
    int gen_seq_len;
    int all_seq_len;
    
    // Performance metrics
    int64_t load_us;
    int64_t vision_us;
    int64_t audio_us;
    int64_t prefill_us;
    int64_t decode_us;
    int64_t sample_us;
    float pixels_mp;
    float audio_input_s;
    
    // Current token info
    int current_token;
    
    // Generated string
    const char* generate_str;
} LLM_Context;

// Create a new LLM instance
MNN_C_API LlmHandle LLM_Create(const char* config_path);

// Destroy an LLM instance
MNN_C_API void LLM_Destroy(LlmHandle handle);

// Load the LLM model
MNN_C_API bool LLM_Load(LlmHandle handle);

// Set configuration
MNN_C_API bool LLM_SetConfig(LlmHandle handle, const char* config_json);

// Get current configuration
MNN_C_API const char* LLM_DumpConfig(LlmHandle handle);

// Generate text
MNN_C_API const char* LLM_Generate(LlmHandle handle, int max_tokens);

// Get response for a given prompt
MNN_C_API const char* LLM_Response(LlmHandle handle, const char* prompt, bool stream);

// Note: Tuning functions are not available in the current LLM implementation

// Forward pass with input
MNN_C_API const char* LLM_Forward(LlmHandle handle, const char* input, const char* image_path, const char* audio_path);

// Reset the LLM state
MNN_C_API void LLM_Reset(LlmHandle handle);

// Check if generation has stopped
MNN_C_API bool LLM_IsStoped(LlmHandle handle);

// Tokenize a string
MNN_C_API int* LLM_Tokenize(LlmHandle handle, const char* text, int* token_count);

// Detokenize tokens
MNN_C_API const char* LLM_Detokenize(LlmHandle handle, const int* tokens, int token_count);

// Get the current context
MNN_C_API LLM_Context* LLM_GetContext(LlmHandle handle);

// Free memory allocated by the API functions
MNN_C_API void LLM_FreeText(const char* text);
MNN_C_API void LLM_FreeTokens(int* tokens);

#ifdef __cplusplus
}
#endif

#endif // LLM_C_H
