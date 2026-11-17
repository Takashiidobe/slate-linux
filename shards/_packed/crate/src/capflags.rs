
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! Defines x86 CPU feature bits
//! Defines VMX CPU feature bits
// This header exists to force full rebuild when the compiler is upgraded.
//
// When fixdep scans this, it will find this string "CONFIG_CC_VERSION_TEXT"
// and add dependency on include/config/CC_VERSION_TEXT, which is touched
// by Kconfig when the version string from the compiler changes.
// Additional tree-wide dependencies start here.
// If any of the GCC plugins change, we need to rebuild everything that
// was built with them, as they may have changed their behavior and those
// behaviors may need to be synchronized across all translation units.
// If the randstruct seed itself changes (whether for GCC plugins or
// Clang), the entire tree needs to be rebuilt since the randomization of
// structures may change between compilation units if not.
// If any external changes affect Clang's integer wrapping sanitizer
// behavior, a full rebuild is needed as the coverage for wrapping types
// may have changed, which may impact the expected behaviors that should
// not differ between compilation units.
// The use of "&&" / "||" is limited in certain expressions.
// The following enable to calculate "and" / "or" with macro expansion only.
// Helper macros to use CONFIG_ options in C/CPP expressions. Note that
// these only work with boolean and tristate options.
// Getting something that works in C and CPP for an arg that may or may
// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
// we match on the placeholder define, insert the "0," for arg1 and generate
// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
// the last step cherry picks the 2nd arg, we get a zero.
// IS_BUILTIN(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y', 0
// otherwise. For boolean options, this is equivalent to
// IS_ENABLED(CONFIG_FOO).
// IS_MODULE(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'm', 0
// otherwise.  CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1" in
// autoconf.h.
// IS_REACHABLE(CONFIG_FOO) evaluates to 1 if the currently compiled
// code can call a function defined in code compiled based on CONFIG_FOO.
// This is similar to IS_ENABLED(), but returns false when invoked from
// built-in code when CONFIG_FOO is set to 'm'.
// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
// __has_builtin is supported on gcc >= 10, clang >= 3 and icc >= 21.
// In the meantime, to support gcc < 10, we implement __has_builtin
// by hand.
// Indirect macros required for expanded argument pasting, eg. __LINE__.
// C23 introduces "auto" as a standard way to define type-inferred
// variables, but "auto" has been a (useless) keyword even since K&R C,
// so it has always been "namespace reserved."
//
// Until at some future time we require C23 support, we need the gcc
// extension __auto_type, but there is no reason to put that elsewhere
// in the source code.
// nothing
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-cold-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Label-Attributes.html#index-cold-label-attribute
//
// When -falign-functions=N is in use, we must avoid the cold attribute as
// GCC drops the alignment for cold functions. Worse, GCC can implicitly mark
// callees of cold functions as cold themselves, so it's not sufficient to add
// __function_aligned here as that will not ensure that callees are correctly
// aligned.
//
// See:
//
//   https://lore.kernel.org/lkml/Y77%2FqVgvaJidFpYt@FVFF77S0Q05N
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88345#c9
// On x86-64 and arm64 targets, __preserve_most changes the calling convention
// of a function to make the code in the caller as unintrusive as possible. This
// convention behaves identically to the C calling convention on how arguments
// and return values are passed, but uses a different set of caller- and callee-
// saved registers.
//
// The purpose is to alleviates the burden of saving and recovering a large
// register set before and after the call in the caller.  This is beneficial for
// rarely taken slow paths, such as error-reporting functions that may be called
// from hot paths.
//
// Note: This may conflict with instrumentation inserted on function entry which
// does not use __preserve_most or equivalent convention (if in assembly). Since
// function tracing assumes the normal C calling convention, where the attribute
// is supported, __preserve_most implies notrace.  It is recommended to restrict
// use of the attribute to functions that should or already disable tracing.
//
// Optional: not supported by gcc.
//
// clang: https://clang.llvm.org/docs/AttributeReference.html#preserve-most
// Compiler specific macros.
// Some architectures need to provide custom definitions of macros provided
// by linux/compiler-*.h, and can do so using asm/compiler.h. We include that
// conditionally rather than using an asm-generic wrapper in order to avoid
// build failures if any C compilation, which will include this file via an
// -include argument in c_flags, occurs prior to the asm-generic wrappers being
// generated.
// it doesn't make sense on ARM (currently the only user of __naked)
// to trace naked functions because then mcount is called without
// stack and frame pointer being set up and there is no chance to
// restore the lr register to the value before mcount was called.
// Prefer gnu_inline, so that extern inline functions do not emit an
// externally visible function. This makes extern inline behave as per gnu89
// semantics rather than c99. This prevents multiple symbol definition errors
// of extern inline functions at link time.
// A lot of inline functions can cause havoc with function tracing.
// gcc provides both __inline__ and __inline as alternate spellings of
// the inline keyword, though the latter is undocumented. New kernel
// code should only use the inline spelling, but some existing code
// uses __inline__. Since we #define inline above, to ensure
// __inline__ has the same semantics, we need this #define.
//
// However, the spelling __inline is strictly reserved for referring
// to the bare keyword.
// GCC does not warn about unused static inline functions for -Wunused-function.
// Suppress the warning in clang as well by using __maybe_unused, but enable it
// for W=2 build. This will allow clang to find unused functions.
// Rather then using noinline to prevent stack consumption, use
// noinline_for_stack instead.  For documentation reasons.
// Use noinline_for_tracing for functions that should not be inlined.
// For tracing reasons.
// Sanitizer helper attributes: Because using __always_inline and
// __no_sanitize_* conflict, provide helper attributes that will either expand
// to __no_sanitize_* in compilation units where instrumentation is enabled
// (__SANITIZE_*__), or __always_inline in compilation units without
// instrumentation (__SANITIZE_*__ undefined).
//
// The assume attribute is used to indicate that a certain condition is
// assumed to be true. If this condition is violated at runtime, the behavior
// is undefined. Compilers may or may not use this indication to generate
// optimized code.
//
// Note that the clang documentation states that optimizers may react
// differently to this attribute, and this may even have a negative
// performance impact. Therefore this attribute should be used with care.
//
// Optional: only supported since gcc >= 13
// Optional: only supported since clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Statement-Attributes.html#index-assume-statement-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#id13
// Optional: only supported since gcc >= 15
// Optional: only supported since clang >= 18
//
//   gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108896
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
//
// __bdos on clang < 19.1.2 can erroneously return 0:
// https://github.com/llvm/llvm-project/pull/110497
//
// __bdos on clang < 19.1.3 can be off by 4:
// https://github.com/llvm/llvm-project/pull/112636
// Runtime track number of objects pointed to by a pointer member for use by
// CONFIG_FORTIFY_SOURCE and CONFIG_UBSAN_BOUNDS.
//
// Optional: only supported since gcc >= 16
// Optional: only supported since clang >= 22
//
//   gcc: https://gcc.gnu.org/pipermail/gcc-patches/2025-April/681727.html
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
// Optional: only supported since gcc >= 15
// Optional: not supported by Clang
//
// gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117178
// Apply __counted_by() when the Endianness matches to increase test coverage.
// This designates the minimum number of elements a passed array parameter must
// have. For example:
//
//     void some_function(u8 param[at_least 7]);
//
// If a caller passes an array with fewer than 7 elements, the compiler will
// emit a warning.
// Section for code which can't be instrumented at all
// The __cpuidle section is used twofold:
//
//  1) the original use -- identifying if a CPU is 'stuck' in idle state based
//     on it's instruction pointer. See cpu_in_idle().
//
//  2) supressing instrumentation around where cpuidle disables RCU; where the
//     function isn't strictly required for #1, this is interchangeable with
//     noinstr.
// The below symbols may be defined for one or more, but not ALL, of the above
// compilers. We don't consider that to be an error, so set them to nothing.
// For example, some of them are for compiler specific plugins.
// Any place that could be marked with the "alloc_size" attribute is also
// a place to be marked with the "malloc" attribute, except those that may
// be performing a _reallocation_, as that may alias the existing pointer.
// For these, use __realloc_size().
// When the size of an allocated object is needed, use the best available
// mechanism to find it. (For cases where sizeof() cannot be used.)
//
// Optional: only supported since gcc >= 12
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Object-Size-Checking.html
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#evaluating-object-size
// Determine if an attribute has been applied to a variable.
// Using __annotated needs to check for __annotated being available,
// or negative tests may fail when annotation cannot be checked. For
// example, see the definition of __is_cstr().
// __flex_counter() - Get pointer to counter member for the given
//                    flexible array, if it was annotated with __counted_by()
// @FAM: Pointer to flexible array member of an addressable struct instance
//
// For example, with:
//
// 	struct foo {
// 		int counter;
// 		short array[] __counted_by(counter);
// 	} *p;
//
// __flex_counter(p->array) will resolve to &p->counter.
//
// Note that Clang may not allow this to be assigned to a separate
// variable; it must be used directly.
//
// If p->array is unannotated, this returns (void *)NULL.
// Some versions of gcc do not mark 'asm goto' volatile:
//
//  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103979
//
// We do it here by hand, because it doesn't hurt.
// Clang has trouble with constraints with multiple
// alternative behaviors ("g" , "rm" and "=rm").
// Use __typeof_unqual__() when available.
// Are two types/vars the same type (ignoring qualifiers)?
// __unqual_scalar_typeof(x) - Declare an unqualified scalar type, leaving
// 			       non-scalar types unchanged.
// __signed_scalar_typeof(x) - Declare a signed scalar type, leaving
// 			       non-scalar types unchanged.
// Is this type a native word size -- useful for atomic operations
// #ifdef __OPTIMIZE__ is only a good approximation; for instance "make
// CFLAGS_foo.o=-Og" defines __OPTIMIZE__, does not elide the conditional code
// and can break compilation with wrong error message(s). Combine with
// -U__OPTIMIZE__ when needed.
// 									 * __noreturn is needed to give the compiler enough			 * information to avoid certain possibly-uninitialized			 * warnings (regardless of the build failing).
// compiletime_assert - break build and emit msg if condition is false
// @condition: a compile-time constant condition to check
// @msg:       a message to emit if condition is false
//
// In tradition of POSIX assert, this macro will break the build if the
// supplied condition is *false*, emitting the supplied error message if the
// compiler has support to do so.
// Helpers for emitting diagnostics in pragmas.
// N 32-bit words worth of info
// N 32-bit bug flags
// Intel-defined CPU features, CPUID level 0x00000001 (EDX), word 0
// "fpu" Onboard FPU
// "vme" Virtual Mode Extensions
// "de" Debugging Extensions
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "msr" Model-Specific Registers
// "pae" Physical Address Extensions
// "mce" Machine Check Exception
// "cx8" CMPXCHG8 instruction
// "apic" Onboard APIC
// "sep" SYSENTER/SYSEXIT
// "mtrr" Memory Type Range Registers
// "pge" Page Global Enable
// "mca" Machine Check Architecture
// "cmov" CMOV instructions (plus FCMOVcc, FCOMI with FPU)
// "pat" Page Attribute Table
// "pse36" 36-bit PSEs
// "pn" Processor serial number
// "clflush" CLFLUSH instruction
// "dts" Debug Store
// "acpi" ACPI via MSR
// "mmx" Multimedia Extensions
// "fxsr" FXSAVE/FXRSTOR, CR4.OSFXSR
// "sse"
// "sse2"
// "ss" CPU self snoop
// "ht" Hyper-Threading
// "tm" Automatic clock control
// "ia64" IA-64 processor
// "pbe" Pending Break Enable
// AMD-defined CPU features, CPUID level 0x80000001, word 1
//
// Don't duplicate feature flags which are redundant with Intel!
// "syscall" SYSCALL/SYSRET
// "mp" MP Capable
// "nx" Execute Disable
// "mmxext" AMD MMX extensions
// "fxsr_opt" FXSAVE/FXRSTOR optimizations
// "pdpe1gb" GB pages
// "rdtscp" RDTSCP
// "lm" Long Mode (x86-64, 64-bit support)
// "3dnowext" AMD 3DNow extensions
// "3dnow" 3DNow
// Transmeta-defined CPU features, CPUID level 0x80860001, word 2
// "recovery" CPU in recovery mode
// "longrun" Longrun power control
// "lrti" LongRun table interface
// Other features, Linux-defined mapping, word 3
//
// This range is used for feature bits which conflict or are synthesized
// "cxmmx" Cyrix MMX extensions
// "k6_mtrr" AMD K6 nonstandard MTRRs
// "cyrix_arr" Cyrix ARRs (= MTRRs)
// "centaur_mcr" Centaur MCRs (= MTRRs)
// Free                                 ( 3*32+ 7)
// "constant_tsc" TSC ticks at a constant rate
// free: was #define X86_FEATURE_UP	( 3*32+ 9) * "up" SMP kernel running on UP
// "art" Always running timer (ART)
// "arch_perfmon" Intel Architectural PerfMon
// "pebs" Precise-Event Based Sampling
// "bts" Branch Trace Store
// "rep_good" REP microcode works well
// "amd_lbr_v2" AMD Last Branch Record Extension Version 2
// "acc_power" AMD Accumulated Power Mechanism
// "nopl" The NOPL (0F 1F) instructions
// "xtopology" CPU topology enum extensions
// "tsc_reliable" TSC is known to be reliable
// "nonstop_tsc" TSC does not stop in C states
// "cpuid" CPU has CPUID instruction itself
// "extd_apicid" Extended APICID (8 bits)
// "amd_dcm" AMD multi-node processor
// "aperfmperf" P-State hardware coordination feedback capability (APERF/MPERF MSRs)
// "rapl" AMD/Hygon RAPL interface
// "nonstop_tsc_s3" TSC doesn't stop in S3 state
// "tsc_known_freq" TSC has known frequency
// Intel-defined CPU features, CPUID level 0x00000001 (ECX), word 4
// "pni" SSE-3
// "pclmulqdq" PCLMULQDQ instruction
// "dtes64" 64-bit Debug Store
// "monitor" MONITOR/MWAIT support
// "ds_cpl" CPL-qualified (filtered) Debug Store
// "vmx" Hardware virtualization
// "smx" Safer Mode eXtensions
// "est" Enhanced SpeedStep
// "tm2" Thermal Monitor 2
// "ssse3" Supplemental SSE-3
// "cid" Context ID
// "sdbg" Silicon Debug
// "fma" Fused multiply-add
// "cx16" CMPXCHG16B instruction
// "xtpr" Send Task Priority Messages
// "pdcm" Perf/Debug Capabilities MSR
// "pcid" Process Context Identifiers
// "dca" Direct Cache Access
// "sse4_1" SSE-4.1
// "sse4_2" SSE-4.2
// "x2apic" X2APIC
// "movbe" MOVBE instruction
// "popcnt" POPCNT instruction
// "tsc_deadline_timer" TSC deadline timer
// "aes" AES instructions
// "xsave" XSAVE/XRSTOR/XSETBV/XGETBV instructions
// "avx" Advanced Vector Extensions
// "f16c" 16-bit FP conversions
// "rdrand" RDRAND instruction
// "hypervisor" Running on a hypervisor
// VIA/Cyrix/Centaur-defined CPU features, CPUID level 0xC0000001, word 5
// "sm2" SM2 algorithm
// "sm2_en" SM2 enabled
// "rng" RNG present (xstore)
// "rng_en" RNG enabled
// "ccs" SM3 + SM4 instructions
// "ccs_en" CCS enabled
// "ace" on-CPU crypto (xcrypt)
// "ace_en" on-CPU crypto enabled
// "ace2" Advanced Cryptography Engine v2
// "ace2_en" ACE v2 enabled
// "phe" PadLock Hash Engine
// "phe_en" PHE enabled
// "pmm" PadLock Montgomery Multiplier
// "pmm_en" PMM enabled
// "rng2" RNG v2
// "rng2_en" RNG2 enabled
// "phe2" PadLock Hash Engine v2
// "phe2_en" PHE2 enabled
// "rsa" Big-number arithmetic
// "rsa_en" RSA enabled
// More extended AMD flags: CPUID level 0x80000001, ECX, word 6
// "lahf_lm" LAHF/SAHF in long mode
// "cmp_legacy" If yes HyperThreading not valid
// "svm" Secure Virtual Machine
// "extapic" Extended APIC space
// "cr8_legacy" CR8 in 32-bit mode
// "abm" Advanced bit manipulation
// "sse4a" SSE-4A
// "misalignsse" Misaligned SSE mode
// "3dnowprefetch" 3DNow prefetch instructions
// "osvw" OS Visible Workaround
// "ibs" Instruction Based Sampling
// "xop" Extended AVX instructions
// "skinit" SKINIT/STGI instructions
// "wdt" Watchdog timer
// "lwp" Light Weight Profiling
// "fma4" 4 operands MAC instructions
// "tce" Translation Cache Extension
// "nodeid_msr" NodeId MSR
// "tbm" Trailing Bit Manipulations
// "topoext" Topology extensions CPUID leafs
// "perfctr_core" Core performance counter extensions
// "perfctr_nb" NB performance counter extensions
// "bpext" Data breakpoint extension
// "ptsc" Performance time-stamp counter
// "perfctr_llc" Last Level Cache performance counter extensions
// "mwaitx" MWAIT extension (MONITORX/MWAITX instructions)
// Auxiliary flags: Linux defined - For features scattered in various
// CPUID levels like 0x6, 0xA etc, word 7.
//
// Reuse free bits when adding new feature flags!
// "ring3mwait" Ring 3 MONITOR/MWAIT instructions
// "cpuid_fault" Intel CPUID faulting
// "cpb" AMD Core Performance Boost
// "epb" IA32_ENERGY_PERF_BIAS support
// "cat_l3" Cache Allocation Technology L3
// "cat_l2" Cache Allocation Technology L2
// "cdp_l3" Code and Data Prioritization L3
// "tdx_host_platform" Platform supports being a TDX host
// "hw_pstate" AMD HW-PState
// "proc_feedback" AMD ProcFeedbackInterface
// "pti" Kernel Page Table Isolation enabled
// "intel_ppin" Intel Processor Inventory Number
// "cdp_l2" Code and Data Prioritization L2
// "ssbd" Speculative Store Bypass Disable
// "mba" Memory Bandwidth Allocation
// "perfmon_v2" AMD Performance Monitoring Version 2
// "ibrs" Indirect Branch Restricted Speculation
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "stibp" Single Thread Indirect Branch Predictors
// "ibrs_enhanced" Enhanced IBRS
// Virtualization flags: Linux defined, word 8
// "tpr_shadow" Intel TPR Shadow
// "flexpriority" Intel FlexPriority
// "ept" Intel Extended Page Table
// "vpid" Intel Virtual Processor ID
// "vmmcall" Prefer VMMCALL to VMCALL
// "ept_ad" Intel Extended Page Table access-dirty bit
// "tdx_guest" Intel Trust Domain Extensions Guest
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "tsc_adjust" TSC adjustment MSR 0x3B
// "sgx" Software Guard Extensions
// "bmi1" 1st group bit manipulation extensions
// "hle" Hardware Lock Elision
// "avx2" AVX2 instructions
// "smep" Supervisor Mode Execution Protection
// "bmi2" 2nd group bit manipulation extensions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "invpcid" Invalidate Processor Context ID
// "rtm" Restricted Transactional Memory
// "cqm" Cache QoS Monitoring
// "mpx" Memory Protection Extension
// "rdt_a" Resource Director Technology Allocation
// "avx512f" AVX-512 Foundation
// "avx512dq" AVX-512 DQ (Double/Quad granular) Instructions
// "rdseed" RDSEED instruction
// "adx" ADCX and ADOX instructions
// "smap" Supervisor Mode Access Prevention
// "avx512ifma" AVX-512 Integer Fused Multiply-Add instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// "intel_pt" Intel Processor Trace
// "avx512pf" AVX-512 Prefetch
// "avx512er" AVX-512 Exponential and Reciprocal
// "avx512cd" AVX-512 Conflict Detection
// "sha_ni" SHA1/SHA256 Instruction Extensions
// "avx512bw" AVX-512 BW (Byte/Word granular) Instructions
// "avx512vl" AVX-512 VL (128/256 Vector Length) Extensions
// Extended state features, CPUID level 0x0000000d:1 (EAX), word 10
// "xsaveopt" XSAVEOPT instruction
// "xsavec" XSAVEC instruction
// "xgetbv1" XGETBV with ECX = 1 instruction
// "xsaves" XSAVES/XRSTORS instructions
// Extended auxiliary flags: Linux defined - for features scattered in various
// CPUID levels like 0xf, etc.
//
// Reuse free bits when adding new feature flags!
// "cqm_llc" LLC QoS if 1
// "cqm_occup_llc" LLC occupancy monitoring
// "cqm_mbm_total" LLC Total MBM monitoring
// "cqm_mbm_local" LLC Local MBM monitoring
// "split_lock_detect" #AC for split lock
// "user_shstk" Shadow stack support for user mode applications
// "avx_vnni" AVX VNNI instructions
// "avx512_bf16" AVX512 BFLOAT16 instructions
// "lass" Linear Address Space Separation
// "fred" Flexible Return and Event Delivery
// "lam" Linear Address Masking
// AMD-defined CPU features, CPUID level 0x80000008 (EBX), word 13
// "clzero" CLZERO instruction
// "irperf" Instructions Retired Count
// "xsaveerptr" Always save/restore FP error pointers
// "rdpru" Read processor register at user level
// "wbnoinvd" WBNOINVD instruction
// "amd_ppin" Protected Processor Inventory Number
// "virt_ssbd" Virtualized Speculative Store Bypass Disable
// "cppc" Collaborative Processor Performance Control
// "brs" Branch Sampling available
// Thermal and Power Management Leaf, CPUID level 0x00000006 (EAX), word 14
// "dtherm" Digital Thermal Sensor
// "ida" Intel Dynamic Acceleration
// "arat" Always Running APIC Timer
// "pln" Intel Power Limit Notification
// "pts" Intel Package Thermal Status
// "hwp" Intel Hardware P-states
// "hwp_notify" HWP Notification
// "hwp_act_window" HWP Activity Window
// "hwp_epp" HWP Energy Perf. Preference
// "hwp_pkg_req" HWP Package Level Request
// "hfi" Hardware Feedback Interface
// AMD SVM Feature Identification, CPUID level 0x8000000a (EDX), word 15
// "npt" Nested Page Table support
// "lbrv" LBR Virtualization support
// "svm_lock" SVM locking MSR
// "nrip_save" SVM next_rip save
// "tsc_scale" TSC scaling support
// "vmcb_clean" VMCB clean bits support
// "flushbyasid" Flush-by-ASID support
// "decodeassists" Decode Assists support
// "pausefilter" Filtered pause intercept
// "pfthreshold" Pause filter threshold
// "avic" Virtual Interrupt Controller
// "v_vmsave_vmload" Virtual VMSAVE VMLOAD
// "vgif" Virtual GIF
// "x2avic" Virtual x2apic
// "v_spec_ctrl" Virtual SPEC_CTRL
// "vnmi" Virtual NMI
// Intel-defined CPU features, CPUID level 0x00000007:0 (ECX), word 16
// "avx512vbmi" AVX512 Vector Bit Manipulation instructions
// "umip" User Mode Instruction Protection
// "pku" Protection Keys for Userspace
// "ospke" OS Protection Keys Enable
// "waitpkg" UMONITOR/UMWAIT/TPAUSE Instructions
// "avx512_vbmi2" Additional AVX512 Vector Bit Manipulation Instructions
// "gfni" Galois Field New Instructions
// "vaes" Vector AES
// "vpclmulqdq" Carry-Less Multiplication Double Quadword
// "avx512_vnni" Vector Neural Network Instructions
// "avx512_bitalg" Support for VPOPCNT[B,W] and VPSHUF-BITQMB instructions
// "tme" Intel Total Memory Encryption
// "avx512_vpopcntdq" POPCNT for vectors of DW/QW
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "bus_lock_detect" Bus Lock detect
// "cldemote" CLDEMOTE instruction
// "movdiri" MOVDIRI instruction
// "movdir64b" MOVDIR64B instruction
// "enqcmd" ENQCMD and ENQCMDS instructions
// "sgx_lc" Software Guard Extensions Launch Control
// Linux-defined word for use with scattered/synthetic bits.
// "overflow_recov" MCA overflow recovery support
// "succor" Uncorrectable error containment and recovery
// "smca" Scalable MCA
// Intel-defined CPU features, CPUID level 0x00000007:0 (EDX), word 18
// "avx512_4vnniw" AVX-512 Neural Network Instructions
// "avx512_4fmaps" AVX-512 Multiply Accumulation Single precision
// "fsrm" Fast Short Rep Mov
// "avx512_vp2intersect" AVX-512 Intersect for D/Q
// "md_clear" VERW clears CPU buffers
// "serialize" SERIALIZE instruction
// "tsxldtrk" TSX Suspend Load Address Tracking
// "pconfig" Intel PCONFIG
// "arch_lbr" Intel ARCH LBR
// "ibt" Indirect Branch Tracking
// "amx_bf16" AMX bf16 Support
// "avx512_fp16" AVX512 FP16
// "amx_tile" AMX tile Support
// "amx_int8" AMX int8 Support
// "flush_l1d" Flush L1D cache
// "arch_capabilities" IA32_ARCH_CAPABILITIES MSR (Intel)
// AMD-defined memory encryption features, CPUID level 0x8000001f (EAX), word 19
// "sme" Secure Memory Encryption
// "sev" Secure Encrypted Virtualization
// "sev_es" Secure Encrypted Virtualization - Encrypted State
// "sev_snp" Secure Encrypted Virtualization - Secure Nested Paging
// "debug_swap" SEV-ES full debug state swap support
// "svsm" SVSM present
// Extended auxiliary flags: Linux defined - for features scattered in various
// CPUID levels like 0x80000022, etc and Linux defined features.
//
// Reuse free bits when adding new feature flags!
// "amd_lbr_pmc_freeze" AMD LBR and PMC Freeze
// BUG word(s)
// "f00f" Intel F00F
// "fdiv" FPU FDIV
// "coma" Cyrix 6x86 coma
// "tlb_mmatch" AMD Erratum 383
// "apic_c1e" AMD Erratum 400
// "11ap" Bad local APIC aka 11AP
// "fxsave_leak" FXSAVE leaks FOP/FIP/FOP
// "clflush_monitor" AAI65, CLFLUSH required before MONITOR
// "sysret_ss_attrs" SYSRET doesn't fix up SS attrs
// "null_seg" Nulling a selector preserves the base
// "swapgs_fence" SWAPGS without input dep on GS
// "monitor" IPI required to wake up remote CPU
// "amd_e400" CPU is among the affected by Erratum 400
// "cpu_meltdown" CPU is affected by meltdown attack and needs kernel page table isolation
// "spectre_v1" CPU is affected by Spectre variant 1 attack with conditional branches
// "spectre_v2" CPU is affected by Spectre variant 2 attack with indirect branches
// "spec_store_bypass" CPU is affected by speculative store bypass attack
// "l1tf" CPU is affected by L1 Terminal Fault
// "mds" CPU is affected by Microarchitectural data sampling
// "msbds_only" CPU is only affected by the  MSDBS variant of BUG_MDS
// "swapgs" CPU is affected by speculation through SWAPGS
// "taa" CPU is affected by TSX Async Abort(TAA)
// "itlb_multihit" CPU may incur MCE during certain page attribute changes
// "srbds" CPU may leak RNG bits if not mitigated
// "mmio_stale_data" CPU is affected by Processor MMIO Stale Data vulnerabilities
// unused, was #define X86_BUG_MMIO_UNKNOWN		X86_BUG(26) "mmio_unknown" CPU is too old and its MMIO Stale Data status is unknown
// "retbleed" CPU is affected by RETBleed
// "eibrs_pbrsb" EIBRS is vulnerable to Post Barrier RSB Predictions
// "smt_rsb" CPU is vulnerable to Cross-Thread Return Address Predictions
// "gds" CPU is affected by Gather Data Sampling
// "tdx_pw_mce" CPU may incur #MC if non-TD software does partial write to TDX private memory
// BUG word 2
// "srso" AMD SRSO bug
// "div0" AMD DIV0 speculation bug
// "rfds" CPU is vulnerable to Register File Data Sampling
// "bhi" CPU is affected by Branch History Injection
// "ibpb_no_ret" IBPB omits return target predictions
// "spectre_v2_user" CPU is affected by Spectre variant 2 attack between user processes
// "old_microcode" CPU has old microcode, it is surely vulnerable to something
// "its" CPU is affected by Indirect Target Selection
// "its_native_only" CPU is affected by ITS, VMX is not affected
// "tsa" CPU is affected by Transient Scheduler Attacks
// "vmscape" CPU is affected by VMSCAPE attacks from guests
// "seamret_invd_vmcs" SEAMRET from P-SEAMLDR clears the current VMCS
// N 32-bit words worth of info
// "vnmi" NMI virtualization
// "preemption_timer" VMX Preemption Timer
// "posted_intr" Posted Interrupts
// EPT/VPID features, scattered to bits 16-23
// "invvpid" INVVPID is supported
// "ept_x_only" EPT entries can be execute only
// "ept_ad" EPT Accessed/Dirty bits
// "ept_1gb" 1GB EPT pages
// "ept_5level" 5-level EPT paging
// Aggregated APIC features 24-27
// "flexpriority" TPR shadow + virt APIC
// "apicv" TPR shadow + APIC reg virt + virt intr delivery + posted interrupts
// VM-Functions, shifted to bits 28-31
// "eptp_switching" EPTP switching (in guest)
// "tsc_offset" Offset hardware TSC when read in guest
// "vtpr" TPR virtualization, a.k.a. TPR shadow
// "mtf" VMX single-step VM-Exits
// Secondary Processor-Based VM-Execution Controls, word 2
// "vapic" Virtualize memory mapped APIC accesses
// "ept" Extended Page Tables, a.k.a. Two-Dimensional Paging
// "vpid" Virtual Processor ID (TLB ASID modifier)
// "unrestricted_guest" Allow Big Real Mode and other "invalid" states
// "vapic_reg" Hardware emulation of reads to the virtual-APIC
// "vid" Evaluation and delivery of pending virtual interrupts
// "ple" Conditionally VM-Exit on PAUSE at CPL0
// "shadow_vmcs" VMREAD/VMWRITE in guest can access shadow VMCS
// "pml" Log dirty pages into buffer
// "ept_violation_ve" Conditionally reflect EPT violations as #VE exceptions
// "ept_mode_based_exec" Enable separate EPT EXEC bits for supervisor vs. user
// "tsc_scaling" Scale hardware TSC when read in guest
// "usr_wait_pause" Enable TPAUSE, UMONITOR, UMWAIT in guest
// "notify_vm_exiting" VM-Exit when no event windows after notify window
// Tertiary Processor-Based VM-Execution Controls, word 3
// "ipi_virt" Enable IPI virtualization
// Skipped when running bindgen due to a libclang issue;
// see https://github.com/rust-lang/rust-bindgen/issues/2244.
// Annotating a function/variable with __retain tells the compiler to place
// the object in its own section and set the flag SHF_GNU_RETAIN. This flag
// instructs the linker to retain the object during garbage-cleanup or LTO
// phases.
//
// Note that the __used macro is also used to prevent functions or data
// being optimized out, but operates at the compiler/IR-level and may still
// allow unintended removal of objects during linking.
//
// Optional: only supported since gcc >= 11, clang >= 13
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-retain-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#retain
#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
#[unsafe(no_mangle)]
static x86_cap_flags: __SlateAlign16<[*const u8; 704]> = __SlateAlign16({
let mut __t0: [*const u8; 704] = unsafe { core::mem::zeroed() };
__t0[0] = (b"fpu\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"vme\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"de\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"pse\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"tsc\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"msr\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"pae\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"mce\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"cx8\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"apic\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"sep\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"mtrr\0".as_ptr() as *mut u8) as *const u8;
__t0[13] = (b"pge\0".as_ptr() as *mut u8) as *const u8;
__t0[14] = (b"mca\0".as_ptr() as *mut u8) as *const u8;
__t0[15] = (b"cmov\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"pat\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"pse36\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"pn\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"clflush\0".as_ptr() as *mut u8) as *const u8;
__t0[21] = (b"dts\0".as_ptr() as *mut u8) as *const u8;
__t0[22] = (b"acpi\0".as_ptr() as *mut u8) as *const u8;
__t0[23] = (b"mmx\0".as_ptr() as *mut u8) as *const u8;
__t0[24] = (b"fxsr\0".as_ptr() as *mut u8) as *const u8;
__t0[25] = (b"sse\0".as_ptr() as *mut u8) as *const u8;
__t0[26] = (b"sse2\0".as_ptr() as *mut u8) as *const u8;
__t0[27] = (b"ss\0".as_ptr() as *mut u8) as *const u8;
__t0[28] = (b"ht\0".as_ptr() as *mut u8) as *const u8;
__t0[29] = (b"tm\0".as_ptr() as *mut u8) as *const u8;
__t0[30] = (b"ia64\0".as_ptr() as *mut u8) as *const u8;
__t0[31] = (b"pbe\0".as_ptr() as *mut u8) as *const u8;
__t0[43] = (b"syscall\0".as_ptr() as *mut u8) as *const u8;
__t0[51] = (b"mp\0".as_ptr() as *mut u8) as *const u8;
__t0[52] = (b"nx\0".as_ptr() as *mut u8) as *const u8;
__t0[54] = (b"mmxext\0".as_ptr() as *mut u8) as *const u8;
__t0[57] = (b"fxsr_opt\0".as_ptr() as *mut u8) as *const u8;
__t0[58] = (b"pdpe1gb\0".as_ptr() as *mut u8) as *const u8;
__t0[59] = (b"rdtscp\0".as_ptr() as *mut u8) as *const u8;
__t0[61] = (b"lm\0".as_ptr() as *mut u8) as *const u8;
__t0[62] = (b"3dnowext\0".as_ptr() as *mut u8) as *const u8;
__t0[63] = (b"3dnow\0".as_ptr() as *mut u8) as *const u8;
__t0[64] = (b"recovery\0".as_ptr() as *mut u8) as *const u8;
__t0[65] = (b"longrun\0".as_ptr() as *mut u8) as *const u8;
__t0[67] = (b"lrti\0".as_ptr() as *mut u8) as *const u8;
__t0[96] = (b"cxmmx\0".as_ptr() as *mut u8) as *const u8;
__t0[97] = (b"k6_mtrr\0".as_ptr() as *mut u8) as *const u8;
__t0[98] = (b"cyrix_arr\0".as_ptr() as *mut u8) as *const u8;
__t0[99] = (b"centaur_mcr\0".as_ptr() as *mut u8) as *const u8;
__t0[104] = (b"constant_tsc\0".as_ptr() as *mut u8) as *const u8;
__t0[106] = (b"art\0".as_ptr() as *mut u8) as *const u8;
__t0[107] = (b"arch_perfmon\0".as_ptr() as *mut u8) as *const u8;
__t0[108] = (b"pebs\0".as_ptr() as *mut u8) as *const u8;
__t0[109] = (b"bts\0".as_ptr() as *mut u8) as *const u8;
__t0[112] = (b"rep_good\0".as_ptr() as *mut u8) as *const u8;
__t0[113] = (b"amd_lbr_v2\0".as_ptr() as *mut u8) as *const u8;
__t0[115] = (b"acc_power\0".as_ptr() as *mut u8) as *const u8;
__t0[116] = (b"nopl\0".as_ptr() as *mut u8) as *const u8;
__t0[118] = (b"xtopology\0".as_ptr() as *mut u8) as *const u8;
__t0[119] = (b"tsc_reliable\0".as_ptr() as *mut u8) as *const u8;
__t0[120] = (b"nonstop_tsc\0".as_ptr() as *mut u8) as *const u8;
__t0[121] = (b"cpuid\0".as_ptr() as *mut u8) as *const u8;
__t0[122] = (b"extd_apicid\0".as_ptr() as *mut u8) as *const u8;
__t0[123] = (b"amd_dcm\0".as_ptr() as *mut u8) as *const u8;
__t0[124] = (b"aperfmperf\0".as_ptr() as *mut u8) as *const u8;
__t0[125] = (b"rapl\0".as_ptr() as *mut u8) as *const u8;
__t0[126] = (b"nonstop_tsc_s3\0".as_ptr() as *mut u8) as *const u8;
__t0[127] = (b"tsc_known_freq\0".as_ptr() as *mut u8) as *const u8;
__t0[128] = (b"pni\0".as_ptr() as *mut u8) as *const u8;
__t0[129] = (b"pclmulqdq\0".as_ptr() as *mut u8) as *const u8;
__t0[130] = (b"dtes64\0".as_ptr() as *mut u8) as *const u8;
__t0[131] = (b"monitor\0".as_ptr() as *mut u8) as *const u8;
__t0[132] = (b"ds_cpl\0".as_ptr() as *mut u8) as *const u8;
__t0[133] = (b"vmx\0".as_ptr() as *mut u8) as *const u8;
__t0[134] = (b"smx\0".as_ptr() as *mut u8) as *const u8;
__t0[135] = (b"est\0".as_ptr() as *mut u8) as *const u8;
__t0[136] = (b"tm2\0".as_ptr() as *mut u8) as *const u8;
__t0[137] = (b"ssse3\0".as_ptr() as *mut u8) as *const u8;
__t0[138] = (b"cid\0".as_ptr() as *mut u8) as *const u8;
__t0[139] = (b"sdbg\0".as_ptr() as *mut u8) as *const u8;
__t0[140] = (b"fma\0".as_ptr() as *mut u8) as *const u8;
__t0[141] = (b"cx16\0".as_ptr() as *mut u8) as *const u8;
__t0[142] = (b"xtpr\0".as_ptr() as *mut u8) as *const u8;
__t0[143] = (b"pdcm\0".as_ptr() as *mut u8) as *const u8;
__t0[145] = (b"pcid\0".as_ptr() as *mut u8) as *const u8;
__t0[146] = (b"dca\0".as_ptr() as *mut u8) as *const u8;
__t0[147] = (b"sse4_1\0".as_ptr() as *mut u8) as *const u8;
__t0[148] = (b"sse4_2\0".as_ptr() as *mut u8) as *const u8;
__t0[149] = (b"x2apic\0".as_ptr() as *mut u8) as *const u8;
__t0[150] = (b"movbe\0".as_ptr() as *mut u8) as *const u8;
__t0[151] = (b"popcnt\0".as_ptr() as *mut u8) as *const u8;
__t0[152] = (b"tsc_deadline_timer\0".as_ptr() as *mut u8) as *const u8;
__t0[153] = (b"aes\0".as_ptr() as *mut u8) as *const u8;
__t0[154] = (b"xsave\0".as_ptr() as *mut u8) as *const u8;
__t0[156] = (b"avx\0".as_ptr() as *mut u8) as *const u8;
__t0[157] = (b"f16c\0".as_ptr() as *mut u8) as *const u8;
__t0[158] = (b"rdrand\0".as_ptr() as *mut u8) as *const u8;
__t0[159] = (b"hypervisor\0".as_ptr() as *mut u8) as *const u8;
__t0[160] = (b"sm2\0".as_ptr() as *mut u8) as *const u8;
__t0[161] = (b"sm2_en\0".as_ptr() as *mut u8) as *const u8;
__t0[162] = (b"rng\0".as_ptr() as *mut u8) as *const u8;
__t0[163] = (b"rng_en\0".as_ptr() as *mut u8) as *const u8;
__t0[164] = (b"ccs\0".as_ptr() as *mut u8) as *const u8;
__t0[165] = (b"ccs_en\0".as_ptr() as *mut u8) as *const u8;
__t0[166] = (b"ace\0".as_ptr() as *mut u8) as *const u8;
__t0[167] = (b"ace_en\0".as_ptr() as *mut u8) as *const u8;
__t0[168] = (b"ace2\0".as_ptr() as *mut u8) as *const u8;
__t0[169] = (b"ace2_en\0".as_ptr() as *mut u8) as *const u8;
__t0[170] = (b"phe\0".as_ptr() as *mut u8) as *const u8;
__t0[171] = (b"phe_en\0".as_ptr() as *mut u8) as *const u8;
__t0[172] = (b"pmm\0".as_ptr() as *mut u8) as *const u8;
__t0[173] = (b"pmm_en\0".as_ptr() as *mut u8) as *const u8;
__t0[182] = (b"rng2\0".as_ptr() as *mut u8) as *const u8;
__t0[183] = (b"rng2_en\0".as_ptr() as *mut u8) as *const u8;
__t0[185] = (b"phe2\0".as_ptr() as *mut u8) as *const u8;
__t0[186] = (b"phe2_en\0".as_ptr() as *mut u8) as *const u8;
__t0[187] = (b"rsa\0".as_ptr() as *mut u8) as *const u8;
__t0[188] = (b"rsa_en\0".as_ptr() as *mut u8) as *const u8;
__t0[192] = (b"lahf_lm\0".as_ptr() as *mut u8) as *const u8;
__t0[193] = (b"cmp_legacy\0".as_ptr() as *mut u8) as *const u8;
__t0[194] = (b"svm\0".as_ptr() as *mut u8) as *const u8;
__t0[195] = (b"extapic\0".as_ptr() as *mut u8) as *const u8;
__t0[196] = (b"cr8_legacy\0".as_ptr() as *mut u8) as *const u8;
__t0[197] = (b"abm\0".as_ptr() as *mut u8) as *const u8;
__t0[198] = (b"sse4a\0".as_ptr() as *mut u8) as *const u8;
__t0[199] = (b"misalignsse\0".as_ptr() as *mut u8) as *const u8;
__t0[200] = (b"3dnowprefetch\0".as_ptr() as *mut u8) as *const u8;
__t0[201] = (b"osvw\0".as_ptr() as *mut u8) as *const u8;
__t0[202] = (b"ibs\0".as_ptr() as *mut u8) as *const u8;
__t0[203] = (b"xop\0".as_ptr() as *mut u8) as *const u8;
__t0[204] = (b"skinit\0".as_ptr() as *mut u8) as *const u8;
__t0[205] = (b"wdt\0".as_ptr() as *mut u8) as *const u8;
__t0[207] = (b"lwp\0".as_ptr() as *mut u8) as *const u8;
__t0[208] = (b"fma4\0".as_ptr() as *mut u8) as *const u8;
__t0[209] = (b"tce\0".as_ptr() as *mut u8) as *const u8;
__t0[211] = (b"nodeid_msr\0".as_ptr() as *mut u8) as *const u8;
__t0[213] = (b"tbm\0".as_ptr() as *mut u8) as *const u8;
__t0[214] = (b"topoext\0".as_ptr() as *mut u8) as *const u8;
__t0[215] = (b"perfctr_core\0".as_ptr() as *mut u8) as *const u8;
__t0[216] = (b"perfctr_nb\0".as_ptr() as *mut u8) as *const u8;
__t0[218] = (b"bpext\0".as_ptr() as *mut u8) as *const u8;
__t0[219] = (b"ptsc\0".as_ptr() as *mut u8) as *const u8;
__t0[220] = (b"perfctr_llc\0".as_ptr() as *mut u8) as *const u8;
__t0[221] = (b"mwaitx\0".as_ptr() as *mut u8) as *const u8;
__t0[224] = (b"ring3mwait\0".as_ptr() as *mut u8) as *const u8;
__t0[225] = (b"cpuid_fault\0".as_ptr() as *mut u8) as *const u8;
__t0[226] = (b"cpb\0".as_ptr() as *mut u8) as *const u8;
__t0[227] = (b"epb\0".as_ptr() as *mut u8) as *const u8;
__t0[228] = (b"cat_l3\0".as_ptr() as *mut u8) as *const u8;
__t0[229] = (b"cat_l2\0".as_ptr() as *mut u8) as *const u8;
__t0[230] = (b"cdp_l3\0".as_ptr() as *mut u8) as *const u8;
__t0[231] = (b"tdx_host_platform\0".as_ptr() as *mut u8) as *const u8;
__t0[232] = (b"hw_pstate\0".as_ptr() as *mut u8) as *const u8;
__t0[233] = (b"proc_feedback\0".as_ptr() as *mut u8) as *const u8;
__t0[235] = (b"pti\0".as_ptr() as *mut u8) as *const u8;
__t0[238] = (b"intel_ppin\0".as_ptr() as *mut u8) as *const u8;
__t0[239] = (b"cdp_l2\0".as_ptr() as *mut u8) as *const u8;
__t0[241] = (b"ssbd\0".as_ptr() as *mut u8) as *const u8;
__t0[242] = (b"mba\0".as_ptr() as *mut u8) as *const u8;
__t0[244] = (b"perfmon_v2\0".as_ptr() as *mut u8) as *const u8;
__t0[249] = (b"ibrs\0".as_ptr() as *mut u8) as *const u8;
__t0[250] = (b"ibpb\0".as_ptr() as *mut u8) as *const u8;
__t0[251] = (b"stibp\0".as_ptr() as *mut u8) as *const u8;
__t0[254] = (b"ibrs_enhanced\0".as_ptr() as *mut u8) as *const u8;
__t0[256] = (b"tpr_shadow\0".as_ptr() as *mut u8) as *const u8;
__t0[257] = (b"flexpriority\0".as_ptr() as *mut u8) as *const u8;
__t0[258] = (b"ept\0".as_ptr() as *mut u8) as *const u8;
__t0[259] = (b"vpid\0".as_ptr() as *mut u8) as *const u8;
__t0[271] = (b"vmmcall\0".as_ptr() as *mut u8) as *const u8;
__t0[273] = (b"ept_ad\0".as_ptr() as *mut u8) as *const u8;
__t0[278] = (b"tdx_guest\0".as_ptr() as *mut u8) as *const u8;
__t0[288] = (b"fsgsbase\0".as_ptr() as *mut u8) as *const u8;
__t0[289] = (b"tsc_adjust\0".as_ptr() as *mut u8) as *const u8;
__t0[290] = (b"sgx\0".as_ptr() as *mut u8) as *const u8;
__t0[291] = (b"bmi1\0".as_ptr() as *mut u8) as *const u8;
__t0[292] = (b"hle\0".as_ptr() as *mut u8) as *const u8;
__t0[293] = (b"avx2\0".as_ptr() as *mut u8) as *const u8;
__t0[295] = (b"smep\0".as_ptr() as *mut u8) as *const u8;
__t0[296] = (b"bmi2\0".as_ptr() as *mut u8) as *const u8;
__t0[297] = (b"erms\0".as_ptr() as *mut u8) as *const u8;
__t0[298] = (b"invpcid\0".as_ptr() as *mut u8) as *const u8;
__t0[299] = (b"rtm\0".as_ptr() as *mut u8) as *const u8;
__t0[300] = (b"cqm\0".as_ptr() as *mut u8) as *const u8;
__t0[302] = (b"mpx\0".as_ptr() as *mut u8) as *const u8;
__t0[303] = (b"rdt_a\0".as_ptr() as *mut u8) as *const u8;
__t0[304] = (b"avx512f\0".as_ptr() as *mut u8) as *const u8;
__t0[305] = (b"avx512dq\0".as_ptr() as *mut u8) as *const u8;
__t0[306] = (b"rdseed\0".as_ptr() as *mut u8) as *const u8;
__t0[307] = (b"adx\0".as_ptr() as *mut u8) as *const u8;
__t0[308] = (b"smap\0".as_ptr() as *mut u8) as *const u8;
__t0[309] = (b"avx512ifma\0".as_ptr() as *mut u8) as *const u8;
__t0[311] = (b"clflushopt\0".as_ptr() as *mut u8) as *const u8;
__t0[312] = (b"clwb\0".as_ptr() as *mut u8) as *const u8;
__t0[313] = (b"intel_pt\0".as_ptr() as *mut u8) as *const u8;
__t0[314] = (b"avx512pf\0".as_ptr() as *mut u8) as *const u8;
__t0[315] = (b"avx512er\0".as_ptr() as *mut u8) as *const u8;
__t0[316] = (b"avx512cd\0".as_ptr() as *mut u8) as *const u8;
__t0[317] = (b"sha_ni\0".as_ptr() as *mut u8) as *const u8;
__t0[318] = (b"avx512bw\0".as_ptr() as *mut u8) as *const u8;
__t0[319] = (b"avx512vl\0".as_ptr() as *mut u8) as *const u8;
__t0[320] = (b"xsaveopt\0".as_ptr() as *mut u8) as *const u8;
__t0[321] = (b"xsavec\0".as_ptr() as *mut u8) as *const u8;
__t0[322] = (b"xgetbv1\0".as_ptr() as *mut u8) as *const u8;
__t0[323] = (b"xsaves\0".as_ptr() as *mut u8) as *const u8;
__t0[352] = (b"cqm_llc\0".as_ptr() as *mut u8) as *const u8;
__t0[353] = (b"cqm_occup_llc\0".as_ptr() as *mut u8) as *const u8;
__t0[354] = (b"cqm_mbm_total\0".as_ptr() as *mut u8) as *const u8;
__t0[355] = (b"cqm_mbm_local\0".as_ptr() as *mut u8) as *const u8;
__t0[358] = (b"split_lock_detect\0".as_ptr() as *mut u8) as *const u8;
__t0[375] = (b"user_shstk\0".as_ptr() as *mut u8) as *const u8;
__t0[388] = (b"avx_vnni\0".as_ptr() as *mut u8) as *const u8;
__t0[389] = (b"avx512_bf16\0".as_ptr() as *mut u8) as *const u8;
__t0[390] = (b"lass\0".as_ptr() as *mut u8) as *const u8;
__t0[401] = (b"fred\0".as_ptr() as *mut u8) as *const u8;
__t0[410] = (b"lam\0".as_ptr() as *mut u8) as *const u8;
__t0[416] = (b"clzero\0".as_ptr() as *mut u8) as *const u8;
__t0[417] = (b"irperf\0".as_ptr() as *mut u8) as *const u8;
__t0[418] = (b"xsaveerptr\0".as_ptr() as *mut u8) as *const u8;
__t0[420] = (b"rdpru\0".as_ptr() as *mut u8) as *const u8;
__t0[425] = (b"wbnoinvd\0".as_ptr() as *mut u8) as *const u8;
__t0[439] = (b"amd_ppin\0".as_ptr() as *mut u8) as *const u8;
__t0[441] = (b"virt_ssbd\0".as_ptr() as *mut u8) as *const u8;
__t0[443] = (b"cppc\0".as_ptr() as *mut u8) as *const u8;
__t0[447] = (b"brs\0".as_ptr() as *mut u8) as *const u8;
__t0[448] = (b"dtherm\0".as_ptr() as *mut u8) as *const u8;
__t0[449] = (b"ida\0".as_ptr() as *mut u8) as *const u8;
__t0[450] = (b"arat\0".as_ptr() as *mut u8) as *const u8;
__t0[452] = (b"pln\0".as_ptr() as *mut u8) as *const u8;
__t0[454] = (b"pts\0".as_ptr() as *mut u8) as *const u8;
__t0[455] = (b"hwp\0".as_ptr() as *mut u8) as *const u8;
__t0[456] = (b"hwp_notify\0".as_ptr() as *mut u8) as *const u8;
__t0[457] = (b"hwp_act_window\0".as_ptr() as *mut u8) as *const u8;
__t0[458] = (b"hwp_epp\0".as_ptr() as *mut u8) as *const u8;
__t0[459] = (b"hwp_pkg_req\0".as_ptr() as *mut u8) as *const u8;
__t0[467] = (b"hfi\0".as_ptr() as *mut u8) as *const u8;
__t0[480] = (b"npt\0".as_ptr() as *mut u8) as *const u8;
__t0[481] = (b"lbrv\0".as_ptr() as *mut u8) as *const u8;
__t0[482] = (b"svm_lock\0".as_ptr() as *mut u8) as *const u8;
__t0[483] = (b"nrip_save\0".as_ptr() as *mut u8) as *const u8;
__t0[484] = (b"tsc_scale\0".as_ptr() as *mut u8) as *const u8;
__t0[485] = (b"vmcb_clean\0".as_ptr() as *mut u8) as *const u8;
__t0[486] = (b"flushbyasid\0".as_ptr() as *mut u8) as *const u8;
__t0[487] = (b"decodeassists\0".as_ptr() as *mut u8) as *const u8;
__t0[490] = (b"pausefilter\0".as_ptr() as *mut u8) as *const u8;
__t0[492] = (b"pfthreshold\0".as_ptr() as *mut u8) as *const u8;
__t0[493] = (b"avic\0".as_ptr() as *mut u8) as *const u8;
__t0[495] = (b"v_vmsave_vmload\0".as_ptr() as *mut u8) as *const u8;
__t0[496] = (b"vgif\0".as_ptr() as *mut u8) as *const u8;
__t0[498] = (b"x2avic\0".as_ptr() as *mut u8) as *const u8;
__t0[500] = (b"v_spec_ctrl\0".as_ptr() as *mut u8) as *const u8;
__t0[505] = (b"vnmi\0".as_ptr() as *mut u8) as *const u8;
__t0[513] = (b"avx512vbmi\0".as_ptr() as *mut u8) as *const u8;
__t0[514] = (b"umip\0".as_ptr() as *mut u8) as *const u8;
__t0[515] = (b"pku\0".as_ptr() as *mut u8) as *const u8;
__t0[516] = (b"ospke\0".as_ptr() as *mut u8) as *const u8;
__t0[517] = (b"waitpkg\0".as_ptr() as *mut u8) as *const u8;
__t0[518] = (b"avx512_vbmi2\0".as_ptr() as *mut u8) as *const u8;
__t0[520] = (b"gfni\0".as_ptr() as *mut u8) as *const u8;
__t0[521] = (b"vaes\0".as_ptr() as *mut u8) as *const u8;
__t0[522] = (b"vpclmulqdq\0".as_ptr() as *mut u8) as *const u8;
__t0[523] = (b"avx512_vnni\0".as_ptr() as *mut u8) as *const u8;
__t0[524] = (b"avx512_bitalg\0".as_ptr() as *mut u8) as *const u8;
__t0[525] = (b"tme\0".as_ptr() as *mut u8) as *const u8;
__t0[526] = (b"avx512_vpopcntdq\0".as_ptr() as *mut u8) as *const u8;
__t0[528] = (b"la57\0".as_ptr() as *mut u8) as *const u8;
__t0[534] = (b"rdpid\0".as_ptr() as *mut u8) as *const u8;
__t0[536] = (b"bus_lock_detect\0".as_ptr() as *mut u8) as *const u8;
__t0[537] = (b"cldemote\0".as_ptr() as *mut u8) as *const u8;
__t0[539] = (b"movdiri\0".as_ptr() as *mut u8) as *const u8;
__t0[540] = (b"movdir64b\0".as_ptr() as *mut u8) as *const u8;
__t0[541] = (b"enqcmd\0".as_ptr() as *mut u8) as *const u8;
__t0[542] = (b"sgx_lc\0".as_ptr() as *mut u8) as *const u8;
__t0[544] = (b"overflow_recov\0".as_ptr() as *mut u8) as *const u8;
__t0[545] = (b"succor\0".as_ptr() as *mut u8) as *const u8;
__t0[547] = (b"smca\0".as_ptr() as *mut u8) as *const u8;
__t0[578] = (b"avx512_4vnniw\0".as_ptr() as *mut u8) as *const u8;
__t0[579] = (b"avx512_4fmaps\0".as_ptr() as *mut u8) as *const u8;
__t0[580] = (b"fsrm\0".as_ptr() as *mut u8) as *const u8;
__t0[584] = (b"avx512_vp2intersect\0".as_ptr() as *mut u8) as *const u8;
__t0[586] = (b"md_clear\0".as_ptr() as *mut u8) as *const u8;
__t0[590] = (b"serialize\0".as_ptr() as *mut u8) as *const u8;
__t0[592] = (b"tsxldtrk\0".as_ptr() as *mut u8) as *const u8;
__t0[594] = (b"pconfig\0".as_ptr() as *mut u8) as *const u8;
__t0[595] = (b"arch_lbr\0".as_ptr() as *mut u8) as *const u8;
__t0[596] = (b"ibt\0".as_ptr() as *mut u8) as *const u8;
__t0[598] = (b"amx_bf16\0".as_ptr() as *mut u8) as *const u8;
__t0[599] = (b"avx512_fp16\0".as_ptr() as *mut u8) as *const u8;
__t0[600] = (b"amx_tile\0".as_ptr() as *mut u8) as *const u8;
__t0[601] = (b"amx_int8\0".as_ptr() as *mut u8) as *const u8;
__t0[604] = (b"flush_l1d\0".as_ptr() as *mut u8) as *const u8;
__t0[605] = (b"arch_capabilities\0".as_ptr() as *mut u8) as *const u8;
__t0[608] = (b"sme\0".as_ptr() as *mut u8) as *const u8;
__t0[609] = (b"sev\0".as_ptr() as *mut u8) as *const u8;
__t0[611] = (b"sev_es\0".as_ptr() as *mut u8) as *const u8;
__t0[612] = (b"sev_snp\0".as_ptr() as *mut u8) as *const u8;
__t0[622] = (b"debug_swap\0".as_ptr() as *mut u8) as *const u8;
__t0[636] = (b"svsm\0".as_ptr() as *mut u8) as *const u8;
__t0[672] = (b"amd_lbr_pmc_freeze\0".as_ptr() as *mut u8) as *const u8;
__t0
});

#[unsafe(no_mangle)]
static x86_bug_flags: __SlateAlign16<[*const u8; 64]> = __SlateAlign16({
let mut __t0: [*const u8; 64] = unsafe { core::mem::zeroed() };
__t0[0] = (b"f00f\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"fdiv\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"coma\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"tlb_mmatch\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"apic_c1e\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"11ap\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"fxsave_leak\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"clflush_monitor\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"sysret_ss_attrs\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"null_seg\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"swapgs_fence\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"monitor\0".as_ptr() as *mut u8) as *const u8;
__t0[13] = (b"amd_e400\0".as_ptr() as *mut u8) as *const u8;
__t0[14] = (b"cpu_meltdown\0".as_ptr() as *mut u8) as *const u8;
__t0[15] = (b"spectre_v1\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"spectre_v2\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"spec_store_bypass\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"l1tf\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"mds\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"msbds_only\0".as_ptr() as *mut u8) as *const u8;
__t0[21] = (b"swapgs\0".as_ptr() as *mut u8) as *const u8;
__t0[22] = (b"taa\0".as_ptr() as *mut u8) as *const u8;
__t0[23] = (b"itlb_multihit\0".as_ptr() as *mut u8) as *const u8;
__t0[24] = (b"srbds\0".as_ptr() as *mut u8) as *const u8;
__t0[25] = (b"mmio_stale_data\0".as_ptr() as *mut u8) as *const u8;
__t0[27] = (b"retbleed\0".as_ptr() as *mut u8) as *const u8;
__t0[28] = (b"eibrs_pbrsb\0".as_ptr() as *mut u8) as *const u8;
__t0[29] = (b"smt_rsb\0".as_ptr() as *mut u8) as *const u8;
__t0[30] = (b"gds\0".as_ptr() as *mut u8) as *const u8;
__t0[31] = (b"tdx_pw_mce\0".as_ptr() as *mut u8) as *const u8;
__t0[32] = (b"srso\0".as_ptr() as *mut u8) as *const u8;
__t0[33] = (b"div0\0".as_ptr() as *mut u8) as *const u8;
__t0[34] = (b"rfds\0".as_ptr() as *mut u8) as *const u8;
__t0[35] = (b"bhi\0".as_ptr() as *mut u8) as *const u8;
__t0[36] = (b"ibpb_no_ret\0".as_ptr() as *mut u8) as *const u8;
__t0[37] = (b"spectre_v2_user\0".as_ptr() as *mut u8) as *const u8;
__t0[38] = (b"old_microcode\0".as_ptr() as *mut u8) as *const u8;
__t0[39] = (b"its\0".as_ptr() as *mut u8) as *const u8;
__t0[40] = (b"its_native_only\0".as_ptr() as *mut u8) as *const u8;
__t0[41] = (b"tsa\0".as_ptr() as *mut u8) as *const u8;
__t0[42] = (b"vmscape\0".as_ptr() as *mut u8) as *const u8;
__t0[43] = (b"seamret_invd_vmcs\0".as_ptr() as *mut u8) as *const u8;
__t0
});

#[unsafe(no_mangle)]
static x86_vmx_flags: __SlateAlign16<[*const u8; 160]> = __SlateAlign16({
let mut __t0: [*const u8; 160] = unsafe { core::mem::zeroed() };
__t0[5] = (b"vnmi\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"preemption_timer\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"posted_intr\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"invvpid\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"ept_x_only\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"ept_ad\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"ept_1gb\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"ept_5level\0".as_ptr() as *mut u8) as *const u8;
__t0[24] = (b"flexpriority\0".as_ptr() as *mut u8) as *const u8;
__t0[25] = (b"apicv\0".as_ptr() as *mut u8) as *const u8;
__t0[28] = (b"eptp_switching\0".as_ptr() as *mut u8) as *const u8;
__t0[35] = (b"tsc_offset\0".as_ptr() as *mut u8) as *const u8;
__t0[53] = (b"vtpr\0".as_ptr() as *mut u8) as *const u8;
__t0[59] = (b"mtf\0".as_ptr() as *mut u8) as *const u8;
__t0[64] = (b"vapic\0".as_ptr() as *mut u8) as *const u8;
__t0[65] = (b"ept\0".as_ptr() as *mut u8) as *const u8;
__t0[69] = (b"vpid\0".as_ptr() as *mut u8) as *const u8;
__t0[71] = (b"unrestricted_guest\0".as_ptr() as *mut u8) as *const u8;
__t0[72] = (b"vapic_reg\0".as_ptr() as *mut u8) as *const u8;
__t0[73] = (b"vid\0".as_ptr() as *mut u8) as *const u8;
__t0[74] = (b"ple\0".as_ptr() as *mut u8) as *const u8;
__t0[78] = (b"shadow_vmcs\0".as_ptr() as *mut u8) as *const u8;
__t0[81] = (b"pml\0".as_ptr() as *mut u8) as *const u8;
__t0[82] = (b"ept_violation_ve\0".as_ptr() as *mut u8) as *const u8;
__t0[86] = (b"ept_mode_based_exec\0".as_ptr() as *mut u8) as *const u8;
__t0[89] = (b"tsc_scaling\0".as_ptr() as *mut u8) as *const u8;
__t0[90] = (b"usr_wait_pause\0".as_ptr() as *mut u8) as *const u8;
__t0[95] = (b"notify_vm_exiting\0".as_ptr() as *mut u8) as *const u8;
__t0[100] = (b"ipi_virt\0".as_ptr() as *mut u8) as *const u8;
__t0
});
