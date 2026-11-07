#include "arg.h"
#include "common.h"
#include "log.h"
#include "betelgeuse-splash.h"

#include "cli-context.h"

#include <filesystem>
#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (cli_context::interrupted().load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    cli_context::interrupted().store(true);
}
#endif

// satisfies -Wmissing-declarations
int llama_cli(int argc, char ** argv);

int llama_cli(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    const char * default_model = "/home/zyxciss/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf";
    const char * default_draft = "/home/zyxciss/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf";
    const bool has_default_model = std::filesystem::exists(default_model);
    if (has_default_model) {
        params.model.path = default_model;
        if (std::filesystem::exists(default_draft)) {
            params.speculative.draft.mparams.path = default_draft;
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
            params.speculative.draft.n_max = 5;
            params.speculative.draft.p_min = 0.50f;
        }
        params.moe_direct_io = true;
        params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        params.speculative.draft.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        params.warmup = false;
        params.fit_params = false;
        params.cache_ram_mib = 0;
    }
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }
    if (has_default_model && params.model.path == default_model) {
        if (params.n_gpu_layers < 0) params.n_gpu_layers = 99;
        if (params.speculative.draft.n_gpu_layers < 0) params.speculative.draft.n_gpu_layers = 99;
        if (params.n_batch == 2048) params.n_batch = 64;
        if (params.n_ubatch == 512) params.n_ubatch = 16;
        if (params.n_ctx == 0) params.n_ctx = 512;
        if (params.cpuparams.n_threads <= 0) params.cpuparams.n_threads = 6;
        if (params.cpuparams_batch.n_threads <= 0) params.cpuparams_batch.n_threads = 6;
    }
    betelgeuse_show_splash(params.use_color);


    llama_backend_init();
    llama_numa_init(params.numa);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    cli_context ctx_cli(params);

    if (!ctx_cli.init()) {
        return 1;
    }

    return ctx_cli.run();
}
