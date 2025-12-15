#include "llm_c.h"
#include "llm/llm.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <sstream>

using namespace MNN::Transformer;

// Get the LLM instance from the handle
std::shared_ptr<Llm> getLlmFromHandle(LlmHandle handle) {
    if (!handle.llm_ptr) {
        return nullptr;
    }
    return *static_cast<std::shared_ptr<Llm>*>(handle.llm_ptr);
}

// Create a new LLM instance
LlmHandle LLM_Create(const char* config_path) {
    LlmHandle handle;
    handle.llm_ptr = nullptr;
    
    try {
        std::shared_ptr<Llm>* llm = new std::shared_ptr<Llm>(Llm::createLLM(config_path));
        handle.llm_ptr = llm;
    } catch (const std::exception& e) {
        std::cerr << "Error creating LLM: " << e.what() << std::endl;
    }
    
    return handle;
}

// Destroy an LLM instance
void LLM_Destroy(LlmHandle handle) {
    if (handle.llm_ptr) {
        delete static_cast<std::shared_ptr<Llm>*>(handle.llm_ptr);
    }
}

// Load the LLM model
bool LLM_Load(LlmHandle handle) {
    auto llm = getLlmFromHandle(handle);
    if (!llm) {
        return false;
    }
    return llm->load();
}

// Set configuration
bool LLM_SetConfig(LlmHandle handle, const char* config_json) {
    auto llm = getLlmFromHandle(handle);
    if (!llm || !config_json) {
        return false;
    }
    return llm->set_config(config_json);
}

// Get current configuration
const char* LLM_DumpConfig(LlmHandle handle) {
    auto llm = getLlmFromHandle(handle);
    if (!llm) {
        return nullptr;
    }
    std::string config = llm->dump_config();
    char* c_config = new char[config.size() + 1];
    std::strcpy(c_config, config.c_str());
    return c_config;
}

// Generate text
const char* LLM_Generate(LlmHandle handle, int max_tokens) {
    auto llm = getLlmFromHandle(handle);
    if (!llm) {
        return nullptr;
    }
    
    llm->generate(max_tokens);
    
    // 获取生成的文本
    const MNN::Transformer::LlmContext* ctx = llm->getContext();
    std::string generated_str = ctx->generate_str;
    char* c_str = new char[generated_str.size() + 1];
    std::strcpy(c_str, generated_str.c_str());
    return c_str;
}

// Get response for a given prompt
const char* LLM_Response(LlmHandle handle, const char* prompt, bool stream) {
    auto llm = getLlmFromHandle(handle);
    if (!llm || !prompt) {
        return nullptr;
    }
    
    std::ostringstream oss;
    // 使用正确的response重载
    llm->response(prompt, &oss, nullptr, stream ? 0 : -1);
    
    std::string response = oss.str();
    char* c_response = new char[response.size() + 1];
    std::strcpy(c_response, response.c_str());
    return c_response;
}

// Forward pass with input
const char* LLM_Forward(LlmHandle handle, const char* input, const char* image_path, const char* audio_path) {
    auto llm = getLlmFromHandle(handle);
    if (!llm || !input) {
        return nullptr;
    }
    
    // 简单实现，仅返回输入
    std::string response(input);
    char* c_response = new char[response.size() + 1];
    std::strcpy(c_response, response.c_str());
    return c_response;
}

// Reset the LLM state
void LLM_Reset(LlmHandle handle) {
    auto llm = getLlmFromHandle(handle);
    if (llm) {
        llm->reset();
    }
}

// Check if generation has stopped
bool LLM_IsStoped(LlmHandle handle) {
    auto llm = getLlmFromHandle(handle);
    if (!llm) {
        return true;
    }
    return llm->stoped();
}

// Tokenize a string
int* LLM_Tokenize(LlmHandle handle, const char* text, int* token_count) {
    auto llm = getLlmFromHandle(handle);
    if (!llm || !text || !token_count) {
        return nullptr;
    }
    
    std::string str_text(text);
    std::vector<int> tokens = llm->tokenizer_encode(str_text);
    
    *token_count = static_cast<int>(tokens.size());
    if (*token_count == 0) {
        return nullptr;
    }
    
    int* c_tokens = new int[*token_count];
    std::copy(tokens.begin(), tokens.end(), c_tokens);
    
    return c_tokens;
}

// Detokenize tokens
const char* LLM_Detokenize(LlmHandle handle, const int* tokens, int token_count) {
    auto llm = getLlmFromHandle(handle);
    if (!llm || !tokens || token_count <= 0) {
        return nullptr;
    }
    
    std::vector<int> vec_tokens(tokens, tokens + token_count);
    
    // 注意：LLM类没有直接的detokenize方法，需要自己实现
    // 这里我们简单地将token转换为字符串，实际应用中需要更复杂的实现
    std::string result;
    for (int token : vec_tokens) {
        result += llm->tokenizer_decode(token);
    }
    
    char* c_result = new char[result.size() + 1];
    std::strcpy(c_result, result.c_str());
    return c_result;
}

// Get the current context
LLM_Context* LLM_GetContext(LlmHandle handle) {
    auto llm = getLlmFromHandle(handle);
    if (!llm) {
        return nullptr;
    }
    
    // 创建一个新的LLM_Context实例
    static LLM_Context c_context;
    const MNN::Transformer::LlmContext* cpp_context = llm->getContext();
    
    // 复制数据
    c_context.prompt_len = cpp_context->prompt_len;
    c_context.gen_seq_len = cpp_context->gen_seq_len;
    c_context.all_seq_len = cpp_context->all_seq_len;
    
    c_context.load_us = cpp_context->load_us;
    c_context.vision_us = cpp_context->vision_us;
    c_context.audio_us = cpp_context->audio_us;
    c_context.prefill_us = cpp_context->prefill_us;
    c_context.decode_us = cpp_context->decode_us;
    c_context.sample_us = cpp_context->sample_us;
    c_context.pixels_mp = cpp_context->pixels_mp;
    c_context.audio_input_s = cpp_context->audio_input_s;
    
    c_context.current_token = cpp_context->current_token;
    c_context.generate_str = cpp_context->generate_str.c_str();
    
    return &c_context;
}

// Free memory allocated by the API functions
void LLM_FreeText(const char* text) {
    if (text) {
        delete[] text;
    }
}

void LLM_FreeTokens(int* tokens) {
    if (tokens) {
        delete[] tokens;
    }
}
