
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! context_unsafe() - disable context checking for contained code
//!
//! Disables context checking for contained statements or expression.
//!
//! .. code-block:: c
//!
//! 	struct some_data {
//! 		spinlock_t lock;
//! 		int counter __guarded_by(&lock);
//! 	};
//!
//! 	int foo(struct some_data *d)
//! 	{
//! 		// ...
//! 		// other code that is still checked ...
//! 		// ...
//! 		return context_unsafe(d->counter);
//! 	}
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! Routines for driver control interface
//! Copyright (c) by Jaroslav Kysela <perex@perex.cz>
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
//! @a is a power of 2 value
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! We need to compute the minimum and maximum values representable in a given
//! type. These macros may also be useful elsewhere. It would seem more obvious
//! to do something like:
//!
//! #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
//! #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
//!
//! Unfortunately, the middle expressions, strictly speaking, have
//! undefined behaviour, and at least some versions of gcc warn about
//! the type_max expression (but not if -fsanitize=undefined is in
//! effect; in that case, the warning is deferred to runtime...).
//!
//! The slightly excessive casting in type_min is to make sure the
//! macros also produce sensible values for the exotic type _Bool. [The
//! overflow checkers only almost work for _Bool, but that's
//! a-feature-not-a-bug, since people shouldn't be doing arithmetic on
//! _Bools. Besides, the gcc builtins don't allow _Bool* as third
//! argument.]
//!
//! Idea stolen from
//! https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
//! credit to Christian Biere.
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! array_index_mask_nospec() - generate a mask that is ~0UL when the
//! 	bounds check succeeds and 0 otherwise
//! @idx: array element index
//! @sz: number of elements in array
//!
//! Returns:
//!     0 - (@idx < @sz)
//! const_*() definitions provide good compile-time optimizations when
//! the passed arguments can be resolved at compile time.
//! Many architecture-specific non-atomic bitops contain inline asm code and due
//! to that the compiler can't optimize them to compile-time expressions or
//! constants. In contrary, generic_*() helpers are defined in pure C and
//! compilers optimize them just well.
//! Therefore, to make `unsigned long foo = 0; __set_bit(BAR, &foo)` effectively
//! equal to `unsigned long foo = BIT(BAR)`, pick the generic C alternative when
//! the arguments can be resolved at compile time. That expression itself is a
//! constant and doesn't bring any functional changes to the rest of cases.
//! The casts to `uintptr_t` are needed to mitigate `-Waddress` warnings when
//! passing a bitmap from .bss or .data (-> `!!addr` is always true).
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These should never be seen by user programs.  To return one of ERESTART*
//! codes, signal_pending() MUST be set.  Note that ptrace can observe these
//! at syscall exit tracing, but they will never be left for the debugged user
//! process to see.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! CR3's layout varies depending on several things.
//!
//! If CR4.PCIDE is set (64-bit only), then CR3[11:0] is the address space ID.
//! If PAE is enabled, then CR3[11:5] is part of the PDPT address
//! (i.e. it's 32-byte aligned, not page-aligned) and CR3[4:0] is ignored.
//! Otherwise (non-PAE, non-PCID), CR3[3] is PWT, CR3[4] is PCD, and
//! CR3[2:0] and CR3[11:5] are ignored.
//!
//! In all cases, Linux puts zeros in the low ignored bits and in PWT and PCD.
//!
//! CR3[63] is always read as zero.  If CR4.PCIDE is set, then CR3[63] may be
//! written as 1 to prevent the write to CR3 from flushing the TLB.
//!
//! On systems with SME, one bit (in a variable position!) is stolen to indicate
//! that the top-level paging structure is encrypted.
//!
//! On systemms with LAM, bits 61 and 62 are used to indicate LAM mode.
//!
//! All of the remaining bits indicate the physical address of the top-level
//! paging structure.
//!
//! CR3_ADDR_MASK is the mask used by read_cr3_pa().
//!
//! Mask off the address space ID and SME encryption bits.
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! writeable
//! 64bit FS base
//! Use alternative patching for paravirt calls:
//! - For replacing an indirect call with a direct one, use the "normal"
//!   ALTERNATIVE() macro with the indirect call as the initial code sequence,
//!   which will be replaced with the related direct call by using the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! - In case the replacement is either a direct call or a short code sequence
//!   depending on a feature bit, the ALTERNATIVE_2() macro is being used.
//!   The indirect call is the initial code sequence again, while the special
//!   code sequence is selected with the specified feature bit. In case the
//!   feature is not active, the direct call is used as above via the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Convenience macros for the sake of set_current_state:
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! Change timeval to jiffies, trying to avoid the
//! most obvious overflows..
//!
//! And some not so obvious.
//!
//! Note that we don't want to return LONG_MAX, because
//! for various timeout reasons we often end up having
//! to wait "jiffies+1" in order to guarantee that we wait
//! at _least_ "jiffies" - so "jiffies+1" had better still
//! be positive.
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! The seqlock seqcount_t interface does not prescribe a precise sequence of
//! read begin/retry/end. For readers, typically there is a call to
//! read_seqcount_begin() and read_seqcount_retry(), however, there are more
//! esoteric cases which do not follow this pattern.
//!
//! As a consequence, we take the following best-effort approach for raw usage
//! via seqcount_t under KCSAN: upon beginning a seq-reader critical section,
//! pessimistically mark the next KCSAN_SEQLOCK_REGION_MAX memory accesses as
//! atomics; if there is a matching read_seqcount_retry() call, no following
//! memory operations are considered atomic. Usage of the seqlock_t interface
//! is not affected.
//! __MAX_NR_ZONES
//! Allocated nodes are mutable until they have been inserted into the tree,
//! at which time they cannot change their type until they have been removed
//! from the tree and an RCU grace period has passed.
//!
//! Removed nodes have their ->parent set to point to themselves.  RCU readers
//! check ->parent before relying on the value that they loaded from the
//! slots array.  This lets us reuse the slots array for the RCU head.
//!
//! Nodes in the tree point to their parent unless bit 0 is set.
//!
//! 64bit sizes
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! page_folio - Converts from page to folio.
//! @p: The page.
//!
//! Every page is part of a folio.  This function cannot be called on a
//! NULL pointer.
//!
//! Context: No reference, nor lock is required on @page.  If the caller
//! does not hold a reference, this call may race with a folio split, so
//! it should re-check the folio still contains this page after gaining
//! a reference on the folio.
//! Return: The folio which contains this page.
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! We do arithmetic on the LRU lists in various places in the code,
//! so it is important to keep the active lists LRU_ACTIVE higher in
//! the array than the corresponding inactive lists, and to keep
//! the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//!
//! This has to be kept in sync with the statistics in zone_stat_item
//! above and the descriptions in vmstat_text in mm/vmstat.c
//! Helper macro to avoid gfp flags if they are the default one
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! array_index_nospec - sanitize an array index after a bounds check
//!
//! For a code sequence like:
//!
//!     if (index < size) {
//!         index = array_index_nospec(index, size);
//!         val = array[index];
//!     }
//!
//! ...if the CPU speculates past the bounds check then
//! array_index_nospec() will clamp the index within the range of [0,
//! size).
//! VMA locks do not behave like most ordinary locks found in the kernel, so we
//! cannot quite have full lockdep tracking in the way we would ideally prefer.
//!
//! Read locks act as shared locks which exclude an exclusive lock being
//! taken. We therefore mark these accordingly on read lock acquire/release.
//!
//! Write locks are acquired exclusively per-VMA, but released in a shared
//! fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
//! that write lock is released.
//!
//! We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
//! is the fact that, of course, we do lockdep-track the mmap lock rwsem which
//! must be held when taking a VMA write lock.
//!
//! We do, however, want to indicate that during either acquisition of a VMA
//! write lock or detachment of a VMA that we require the lock held be exclusive,
//! so we utilise lockdep to do so.
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! 6 because 6 should be plenty and struct tlb_state will fit in two cache
//! lines.
//! This is the smallest unsigned integer type that can fit a value
//! (up to 'long long')
//! CAP_BPF allows the following BPF operations:
//! - Creating all types of BPF maps
//! - Advanced verifier features
//!   - Indirect variable access
//!   - Bounded loops
//!   - BPF to BPF function calls
//!   - Scalar precision tracking
//!   - Larger complexity limits
//!   - Dead code elimination
//!   - And potentially other features
//! - Loading BPF Type Format (BTF) data
//! - Retrieve xlated and JITed code of BPF programs
//! - Use bpf_spin_lock() helper
//!
//! CAP_PERFMON relaxes the verifier checks further:
//! - BPF progs can use of pointer-to-integer conversions
//! - speculation attack hardening measures are bypassed
//! - bpf_probe_read to read arbitrary kernel memory is allowed
//! - bpf_trace_printk to print kernel memory is allowed
//!
//! CAP_SYS_ADMIN is required to use bpf_probe_write_user.
//!
//! CAP_SYS_ADMIN is required to iterate system wide loaded
//! programs, maps, links, BTFs and convert their IDs to file descriptors.
//!
//! CAP_PERFMON and CAP_BPF are required to load tracing programs.
//! CAP_NET_ADMIN and CAP_BPF are required to load networking programs.
//! return the first ptr or next element in an RCU protected list
//! This hash multiplies the input by a large odd number and takes the
//! high bits.  Since multiplication propagates changes to the most
//! significant end only, it is essential that the high bits of the
//! product be used for the hash value.
//!
//! Chuck Lever verified the effectiveness of this technique:
//! http://www.citi.umich.edu/techreports/reports/citi-tr-00-1.pdf
//!
//! Although a random odd number will do, it turns out that the golden
//! ratio phi = (sqrt(5)-1)/2, or its negative, has particularly nice
//! properties.  (See Knuth vol 3, section 6.4, exercise 9.)
//!
//! These are the negative, (1 - phi) = phi**2 = (3 - sqrt(5))/2,
//! which is very slightly easier to multiply by and makes no
//! difference to the hash distribution.
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! The structure layout of user_xstateregs, used for exporting the
//! extended register state through ptrace and core-dump (NT_X86_XSTATE note)
//! interfaces will be same as the memory layout of xsave used by the processor
//! (except for the bytes 464..511, which can be used by the software) and hence
//! the size of this structure varies depending on the features supported by the
//! processor and OS. The size of the structure that users need to use can be
//! obtained by doing:
//!     cpuid_count(0xd, 0, &eax, &ptrace_xstateregs_struct_size, &ecx, &edx);
//! i.e., cpuid.(eax=0xd,ecx=0).ebx will be the size that user (debuggers, etc.)
//! need to use.
//!
//! For now, only the first 8 bytes of the software usable bytes[464..471] will
//! be used and will be set to OS enabled xstate mask (which is same as the
//! 64bit mask returned by the xgetbv's xCR0).  Users (analyzing core dump
//! remotely, etc.) can use this mask as well as the mask saved in the
//! xstate_hdr bytes and interpret what states the processor/OS supports
//! and what states are in modified/initialized conditions for the
//! particular process/thread.
//!
//! Also when the user modifies certain state FP/SSE/etc through the
//! ptrace interface, they must ensure that the header.xfeatures
//! bytes[512..519] of the memory layout are updated correspondingly.
//! i.e., for example when FP state is modified to a non-init state,
//! header.xfeatures's bit 0 must be set to '1', when SSE is modified to
//! non-init state, header.xfeatures's bit 1 must to be set to '1', etc.
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generic info of form tag = "info"
//! ***DANGEROUS*** don't add guard page
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! Generate SD flag indexes
//! don't change - minor numbers
//! same as first respective minor number to make minor allocation easier
//! These are specified by iBCS2
//! Epoll event masks
//! kernel three bits - LED group
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
// __context_unsafe() - function attribute, disable context checking
// @comment: comment explaining why opt-out is safe
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Forces adding an inline comment as argument.
// token_context_lock() - declare an abstract global context lock instance
// @name: token context lock name
//
// Helper that declares an abstract global context lock instance @name, but not
// backed by a real data structure (linker error if accidentally referenced).
// The type name is `__ctx_lock_@name`.
// token_context_lock_instance() - declare another instance of a global context lock
// @ctx: token context lock previously declared with token_context_lock()
// @name: name of additional global context lock instance
//
// Helper that declares an additional instance @name of the same token context
// lock class @ctx. This is helpful where multiple related token contexts are
// declared, to allow using the same underlying type (`__ctx_lock_@ctx`) as
// function arguments.
// __must_hold() - function attribute, caller must hold exclusive context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) exclusively.
// __acquires() - function attribute, function acquires context lock exclusively
//
// Function attribute declaring that the function acquires the given context
// lock instance(s) exclusively, but does not release them.
// __cond_acquires() - function attribute, function conditionally
//                     acquires a context lock exclusively
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x exclusively, but does not release it. The
// function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases() - function attribute, function releases a context lock exclusively
//
// Function attribute declaring that the function releases the given context
// lock instance(s) exclusively. The associated context(s) must be active on
// entry.
// __must_hold_shared() - function attribute, caller must hold shared context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) with shared access.
// __acquires_shared() - function attribute, function acquires context lock shared
//
// Function attribute declaring that the function acquires the given
// context lock instance(s) with shared access, but does not release them.
// __cond_acquires_shared() - function attribute, function conditionally
//                            acquires a context lock shared
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x with shared access, but does not release it.
// The function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases_shared() - function attribute, function releases a
//                       context lock shared
//
// Function attribute declaring that the function releases the given context
// lock instance(s) with shared access. The associated context(s) must be
// active on entry.
// __acquire_shared() - function to acquire context lock shared
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x with shared
// access.
// __release_shared() - function to release context lock shared
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x with shared
// access.
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
// Note: do not use this directly. Instead, use __alloc_size() since it is conditionally
// available and includes other attributes. For GCC < 9.1, __alloc_size__ gets undefined
// in compiler-gcc.h, due to misbehaviors.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alloc_005fsize-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#alloc-size
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
// The second argument is optional (default 0), so we use a variadic macro
// to make the shorthand.
//
// Beware: Do not apply this to functions which may return
// ERR_PTRs. Also, it is probably unwise to apply it to functions
// returning extra information in the low bits (but in that case the
// compiler should see some alignment anyway, when the return value is
// massaged by 'flags = ptr & 3; ptr &= ~3;').
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-assume_005faligned-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#assume-aligned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-cleanup-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#cleanup
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
// Note: deliberately not named '__nonnull', to avoid clashing with glibc's
// __nonnull() when kernel and userspace headers are combined.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html#index-nonnull
// clang: https://clang.llvm.org/docs/AttributeReference.html#nonnull
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noreturn-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#noreturn
// clang: https://clang.llvm.org/docs/AttributeReference.html#id1
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-section-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-section-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-warn_005funused_005fresult-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#nodiscard-warn-unused-result
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
// This version is i.e. to prevent dead stores elimination on @ptr
// where gcc and llvm may behave differently when otherwise using
// normal barrier(): while gcc behavior gets along with a normal
// barrier(), llvm needs an explicit input variable to be assumed
// clobbered. The issue is as follows: while the inline asm might
// access any memory it wants, the compiler could have fit all of
// @ptr into memory registers instead, and since @ptr never escaped
// from that, it proved that the inline asm wasn't touching any of
// it. This version works well with both compilers, i.e. we're telling
// the compiler that the inline asm absolutely may see the contents
// of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// data_race - mark an expression as containing intentional data races
//
// This data_race() macro is useful for situations in which data races
// should be forgiven.  One example is diagnostic code that accesses
// shared variables but is not a part of the core synchronization design.
// For example, if accesses to a given variable are protected by a lock,
// except for diagnostic code, then the accesses under the lock should
// be plain C-language accesses and those in the diagnostic code should
// use data_race().  This way, KCSAN will complain if buggy lockless
// accesses to that variable are introduced, even if the buggy accesses
// are protected by READ_ONCE() or WRITE_ONCE().
//
// This macro *does not* affect normal code generation, but is a hint
// to tooling that data races here are to be ignored.  If the access must
// be atomic *and* KCSAN should ignore the access, use both data_race()
// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
// &a[0] degrades to a pointer: a different type from an array
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// This returns a constant expression while determining if an argument is
// a constant expression, most importantly without evaluating the argument.
// Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//
// Details:
// - sizeof() return an integer constant expression, and does not evaluate
//   the value of its operand; it only examines the type of its operand.
// - The results of comparing two integer constant expressions is also
//   an integer constant expression.
// - The first literal "8" isn't important. It could be any literal value.
// - The second literal "8" is to avoid warnings about unaligned pointers;
//   this could otherwise just be "1".
// - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//   architectures.
// - The C Standard defines "null pointer constant", "(void *)0", as
//   distinct from other void pointers.
// - If (x) is an integer constant expression, then the "* 0l" resolves
//   it into an integer constant expression of value 0. Since it is cast to
//   "void *", this makes the second operand a null pointer constant.
// - If (x) is not an integer constant expression, then the second operand
//   resolves to a void pointer (but not a null pointer constant: the value
//   is not an integer constant 0).
// - The conditional operator's third operand, "(int *)8", is an object
//   pointer (to type "int").
// - The behavior (including the return type) of the conditional operator
//   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//   given for the second and third operands. This is the central mechanism
//   of the macro:
//   - When one operand is a null pointer constant (i.e. when x is an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns the type of the
//     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//     would then get:
//       sizeof(*((int *)(...))  == sizeof(int)  == 4
//   - When one operand is a void pointer (i.e. when x is not an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns a "void *" type.
//     Here, within the sizeof(), we would then get:
//       sizeof(*((void *)(...)) == sizeof(void) == 1
// - The equality comparison to "sizeof(int)" therefore depends on (x):
//     sizeof(int) == sizeof(int)     (x) was a constant expression
//     sizeof(int) != sizeof(void)    (x) was not a constant expression
// Similar to statically_true() but produces a constant expression
//
// To be used in conjunction with macros, such as BUILD_BUG_ON_ZERO(),
// which require their input to be a constant expression and for which
// statically_true() would otherwise fail.
//
// This is a trade-off: const_true() requires all its operands to be
// compile time constants. Else, it would always returns false even on
// the most trivial cases like:
//
//   true || non_const_var
//
// On the opposite, statically_true() is able to fold more complex
// tautologies and will return true on expressions such as:
//
//   !(non_const_var * 8 % 4)
//
// For the general case, statically_true() is better.
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
// expression but avoids the generation of any code, even if that expression
// has side-effects.
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
// BUILD_BUG_ON - break compile if a condition is true.
// @condition: the condition which the compiler should know is false.
//
// If you have some code which relies on certain constants being equal, or
// some other compile-time-evaluated condition, you should use BUILD_BUG_ON to
// detect if someone changes it.
// BUILD_BUG - break compile if used.
//
// If you have some code that you expect the compiler to eliminate at
// build time, you should use BUILD_BUG to detect if it is
// unexpectedly used.
// static_assert - check integer constant expression at build time
// @expr: expression to be checked
//
// static_assert() is a wrapper for the C11 _Static_assert, with a
// little macro magic to make the message optional (defaulting to the
// stringification of the tested expression).
//
// Contrary to BUILD_BUG_ON(), static_assert() can be used at global
// scope, but requires the expression to be an integer constant
// expression (i.e., it is not enough that __builtin_constant_p() is
// true for expr).
//
// Also note that BUILD_BUG_ON() fails the build if the condition is
// true, while static_assert() fails the build if the expression is
// false.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
// Because instrumentation_{begin,end}() can nest, objtool validation considers
// _begin() a +1 and _end() a -1 and computes a sum over the instructions.
// When the value is greater than 0, we consider instrumentation allowed.
//
// There is a problem with code like:
//
// noinstr void foo()
// {
// 	instrumentation_begin();
// 	...
// 	if (cond) {
// 		instrumentation_begin();
// 		...
// 		instrumentation_end();
// 	}
// 	bar();
// 	instrumentation_end();
// }
//
// If instrumentation_end() would be an empty label, like all the other
// annotations, the inner _end(), which is at the end of a conditional block,
// would land on the instruction after the block.
//
// If we then consider the sum of the !cond path, we'll see that the call to
// bar() is with a 0-value, even though, we meant it to happen with a positive
// value.
//
// To avoid this, have _end() be a NOP instruction, this ensures it will be
// part of the condition block and does not escape.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Some subsystems have their own custom printk that applies a va_format to a
// generic format, for example, to include a device number or other metadata
// alongside the format supplied by the caller.
//
// In order to store these in the way they would be emitted by the printk
// infrastructure, the subsystem provides us with the start, fixed string, and
// any subsequent text in the format string.
//
// We take a variable argument list as pr_fmt/dev_fmt/etc are sometimes passed
// as multiple arguments (eg: `"%s: ", "blah"`), and we must only take the
// first one.
//
// subsys_fmt_prefix must be known at compile time, or compilation will fail
// (since this is a mistake). If fmt or level is not known at compile time, no
// index entry will be made (since this can legitimately happen).
// printk - print a kernel message
// @fmt: format string
//
// This is printk(). It can be called from any context. We want it to work.
//
// If printk indexing is enabled, _printk() is called from printk_index_wrap.
// Otherwise, printk is simply #defined to _printk.
//
// We try to grab the console_lock. If we succeed, it's easy - we log the
// output and call the console drivers.  If we fail to get the semaphore, we
// place the output into the log buffer and return. The current holder of
// the console_sem will notice the new output in console_unlock(); and will
// send it to the consoles before releasing the lock.
//
// One effect of this deferred printing is that code which calls printk() and
// then changes console_loglevel may break. This is because console_loglevel
// is inspected when the actual printing occurs.
//
// See also:
// printf(3)
//
// See the vsnprintf() documentation for format string extensions over C99.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// If @feature is set, patch in @newinstr_yes, otherwise @newinstr_no.
// Alternative instructions for different CPU types or capabilities.
//
// This allows to use optimized instructions even on generic binary
// kernels.
//
// length of oldinstr must be longer or equal the length of newinstr
// It can be padded with nops as needed.
//
// For non barrier like inlines please define new variants
// without volatile and memory clobber.
// Alternative inline assembly with input.
//
// Peculiarities:
// No memory clobber here.
// Argument numbers start with 1.
// Leaving an unused argument 0 to keep API compatibility.
// Like alternative_input, but with a single output argument
// Prevent speculative execution past this barrier.
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "sse"
// "pdpe1gb" GB pages
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// order_base_2 - calculate the (rounded up) base 2 order of the argument
// @n: parameter
//
// The first few values calculated by this routine:
//  ob2(0) = 0
//  ob2(1) = 0
//  ob2(2) = 1
//  ob2(3) = 2
//  ob2(4) = 2
//  ob2(5) = 3
//  ... and so on.
// bits_per - calculate the number of bits required for the argument
// @n: parameter
//
// This is constant-capable and can be used for compile time
// initializations, e.g bitfields.
//
// The first few values calculated by this routine:
// bf(0) = 1
// bf(1) = 1
// bf(2) = 2
// bf(3) = 2
// bf(4) = 3
// ... and so on.
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
// abs - return absolute value of an argument
// @x: the value.
//
// If it is unsigned type, @x is converted to signed type first.
// char is treated as if it was signed (regardless of whether it really is)
// but the macro's return type is preserved as char.
//
// NOTE, for signed type if @x is the minimum, the returned result is undefined
// as there is not enough bits to represent it as a positive number.
//
// Return: an absolute value of @x.
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// File descriptor in bad state
// Protocol not available
// Operation not supported on transport endpoint
// Memory page has hardware error
// Generate a per-CPU add to memory instruction and optimize code
// if one is added or subtracted.
// Add return operation
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
// alpha modules require percpu variables to be defined as
// weak to force the compiler to generate GOT based external
// references for them.  This is necessary because percpu sections
// will be located outside of the usually addressable area.
//
// This definition puts the following two extra restrictions when
// defining percpu variables.
//
// 1. The symbol must be globally unique, even the static ones.
// 2. Static percpu variables cannot be defined inside a function.
//
// Archs which need weak percpu definitions should set
// CONFIG_ARCH_MODULE_NEEDS_WEAK_PER_CPU when necessary.
//
// Normal declaration and definition macros.
// Variant on the per-CPU variable declaration/definition theme used for
// ordinary per-CPU variables.
// Declaration/definition used for per-CPU variables that are frequently
// accessed and should be in a single cacheline.
//
// For use only by architecture and core code.  Only use scalar or pointer
// types to maximize density.
// Declaration/definition used for per-CPU variables that must be cacheline
// aligned under SMP conditions so that, whilst a particular instance of the
// data corresponds to a particular CPU, inefficiencies due to direct access by
// other CPUs are reduced by preventing the data from unnecessarily spanning
// cachelines.
//
// An example of this would be statistical data, where each CPU's set of data
// is updated by that CPU alone, but the data from across all CPUs is collated
// by a CPU processing a read from a proc file.
// Declaration/definition used for per-CPU variables that must be page aligned.
// Declaration/definition used for per-CPU variables that must be read mostly.
// Add an offset to a pointer.  Use RELOC_HIDE() to prevent the compiler
// from making incorrect assumptions about the pointer value.
// Operations for contexts where we do not want to do any checks for
// preemptions.  Unless strictly necessary, always use [__]this_cpu_*()
// instead.
//
// If there is no other protection through preempt disable and/or disabling
// interrupts then one of these RMW operations can show unexpected behavior
// because the execution thread was rescheduled on another processor or an
// interrupt occurred and the same percpu variable was modified from the
// interrupt context.
// Operations for contexts that are safe from preemption/interrupts.  These
// operations verify that preemption is disabled.
// Operations with implied preemption/interrupt protection.  These
// operations can be used without worrying about preemption or interrupt.
// The preempt_count offset needed for things like:
//
//  spin_lock_bh()
//
// Which need to disable both preemption (CONFIG_PREEMPT_COUNT) and
// softirqs, such that unlock sequences of:
//
//  spin_unlock();
//  local_bh_enable();
//
// Work as expected.
// An exchange-type operation, which takes a value and a pointer, and
// returns the old value.
// Note: no "lock" prefix even on SMP: xchg always implies lock anyway.
// Since this is generally used to protect other memory information, we
// use "asm volatile" and "memory" clobbers to prevent gcc from moving
// information around.
// Atomic compare and exchange.  Compare OLD with MEM, if identical,
// store NEW in MEM.  Return the initial value in MEM.  Success is
// indicated by comparing RETURN with OLD.
// xadd() adds "inc" to "*ptr" and atomically returns the previous
// value of "*ptr".
//
// xadd() is locked when multiple CPUs are online
// No ioctl command
// restart by calling sys_restart_syscall
// Operation is not supported
// strscpy - Copy a C-string into a sized buffer
// @dst: Where to copy the string to
// @src: Where to copy the string from
// @...: Size of destination buffer (optional)
//
// Copy the source string @src, or as much of it as fits, into the
// destination @dst buffer. The behavior is undefined if the string
// buffers overlap. The destination @dst buffer is always NUL terminated,
// unless it's zero-sized.
//
// The size argument @... is only required when @dst is not an array, or
// when the copy needs to be smaller than sizeof(@dst).
//
// Preferred to strncpy() since it always returns a valid string, and
// doesn't unnecessarily force the tail of the destination buffer to be
// zero padded. If padding is desired please use strscpy_pad().
//
// Returns the number of characters copied in @dst (not including the
// trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
// truncated.
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
// ZONE_MOVABLE allowed
// DOC: Page mobility and placement hints
//
// Page mobility and placement hints
//
// These flags provide hints about how mobile the page is. Pages with similar
// mobility are placed within the same pageblocks to minimise problems due
// to external fragmentation.
//
// %__GFP_MOVABLE (also a zone modifier) indicates that the page can be
// moved by page migration during memory compaction or can be reclaimed.
//
// %__GFP_RECLAIMABLE is used for slab allocations that specify
// SLAB_RECLAIM_ACCOUNT and whose pages can be freed via shrinkers.
//
// %__GFP_WRITE indicates the caller intends to dirty the page. Where possible,
// these pages will be spread between local zones to avoid all the dirty
// pages being in one zone (fair zone allocation policy).
//
// %__GFP_HARDWALL enforces the cpuset memory allocation policy.
//
// %__GFP_THISNODE forces the allocation to be satisfied from the requested
// node with no fallbacks or placement policy enforcements.
//
// %__GFP_ACCOUNT causes the allocation to be accounted to the active
// cgroup context.
// DOC: Reclaim modifiers
//
// Reclaim modifiers
// Please note that all the following flags are only applicable to sleepable
// allocations (e.g. %GFP_NOWAIT and %GFP_ATOMIC will ignore them).
//
// %__GFP_IO can start physical IO.
//
// %__GFP_FS can call down to the low-level FS. Clearing the flag avoids the
// allocator recursing into the filesystem which might already be holding
// locks.
//
// %__GFP_DIRECT_RECLAIM indicates that the caller may enter direct reclaim.
// This flag can be cleared to avoid unnecessary delays when a fallback
// option is available.
//
// %__GFP_KSWAPD_RECLAIM indicates that the caller wants to wake kswapd when
// the low watermark is reached and have it reclaim pages until the high
// watermark is reached. A caller may wish to clear this flag when fallback
// options are available and the reclaim is likely to disrupt the system. The
// canonical example is THP allocation where a fallback is cheap but
// reclaim/compaction may cause indirect stalls.
//
// %__GFP_RECLAIM is shorthand to allow/forbid both direct and kswapd reclaim.
//
// The default allocator behavior depends on the request size. We have a concept
// of so-called costly allocations (with order > %PAGE_ALLOC_COSTLY_ORDER).
// !costly allocations are too essential to fail so they are implicitly
// non-failing by default (with some exceptions like OOM victims might fail so
// the caller still has to check for failures) while costly requests try to be
// not disruptive and back off even without invoking the OOM killer.
// The following three modifiers might be used to override some of these
// implicit rules. Please note that all of them must be used along with
// %__GFP_DIRECT_RECLAIM flag.
//
// %__GFP_NORETRY: The VM implementation will try only very lightweight
// memory direct reclaim to get some memory under memory pressure (thus
// it can sleep). It will avoid disruptive actions like OOM killer. The
// caller must handle the failure which is quite likely to happen under
// heavy memory pressure. The flag is suitable when failure can easily be
// handled at small cost, such as reduced throughput.
//
// %__GFP_RETRY_MAYFAIL: The VM implementation will retry memory reclaim
// procedures that have previously failed if there is some indication
// that progress has been made elsewhere.  It can wait for other
// tasks to attempt high-level approaches to freeing memory such as
// compaction (which removes fragmentation) and page-out.
// There is still a definite limit to the number of retries, but it is
// a larger limit than with %__GFP_NORETRY.
// Allocations with this flag may fail, but only when there is
// genuinely little unused memory. While these allocations do not
// directly trigger the OOM killer, their failure indicates that
// the system is likely to need to use the OOM killer soon.  The
// caller must handle failure, but can reasonably do so by failing
// a higher-level request, or completing it only in a much less
// efficient manner.
// If the allocation does fail, and the caller is in a position to
// free some non-essential memory, doing so could benefit the system
// as a whole.
//
// %__GFP_NOFAIL: The VM implementation _must_ retry infinitely: the caller
// cannot handle allocation failures. The allocation could block
// indefinitely but will never return with failure. Testing for
// failure is pointless.
// It _must_ be blockable and used together with __GFP_DIRECT_RECLAIM.
// It should _never_ be used in non-sleepable contexts.
// New users should be evaluated carefully (and the flag should be
// used only when there is no reasonable failure policy) but it is
// definitely preferable to use the flag rather than opencode endless
// loop around allocator.
// Allocating pages from the buddy with __GFP_NOFAIL and order > 1 is
// not supported. Please consider using kvmalloc() instead.
// Caller can reclaim
// DOC: Action modifiers
//
// Action modifiers
//
// %__GFP_NOWARN suppresses allocation failure reports.
//
// %__GFP_COMP address compound page metadata.
//
// %__GFP_ZERO returns a zeroed page on success.
//
// %__GFP_ZEROTAGS zeroes memory tags at allocation time. Setting memory tags at
// the same time as zeroing memory (e.g., with __GFP_ZERO) has minimal
// additional performance impact. However, __GFP_ZEROTAGS also zeroes the tags
// even if memory is not getting zeroed at allocation time (e.g.,
// with init_on_free).
//
// %__GFP_SKIP_KASAN makes KASAN skip unpoisoning on page allocation.
// Used for userspace and vmalloc pages; the latter are unpoisoned by
// kasan_unpoison_vmalloc instead. If passed to vmalloc, kasan_unpoison_vmalloc
// is skipped too. For userspace pages, results in poisoning being skipped as
// well, see should_skip_kasan_poison for details. Only effective in HW_TAGS mode.
// Disable lockdep for GFP context tracking
// DOC: Useful GFP flag combinations
//
// Useful GFP flag combinations
//
// Useful GFP flag combinations that are commonly used. It is recommended
// that subsystems start with one of these combinations and then set/clear
// %__GFP_FOO flags as necessary.
//
// %GFP_ATOMIC users can not sleep and need the allocation to succeed. A lower
// watermark is applied to allow access to "atomic reserves".
// The current implementation doesn't support NMI, nor contexts that disable
// preemption under PREEMPT_RT. This includes raw_spin_lock() and plain
// preempt_disable() - see "Memory allocation" in
// Documentation/core-api/real-time/differences.rst for more info.
//
// %GFP_KERNEL is typical for kernel-internal allocations. The caller requires
// %ZONE_NORMAL or a lower zone for direct access but can direct reclaim.
//
// %GFP_KERNEL_ACCOUNT is the same as GFP_KERNEL, except the allocation is
// accounted to the active cgroup context.
//
// %GFP_NOWAIT is for kernel allocations that should not stall for direct
// reclaim, start physical IO or use any filesystem callback.  It is very
// likely to fail to allocate memory, even for very small allocations.
// The same restrictions on calling contexts apply as for %GFP_ATOMIC.
//
// %GFP_NOIO will use direct reclaim to discard clean pages or slab pages
// that do not require the starting of any physical IO.
// Please try to avoid using this flag directly and instead use
// memalloc_noio_{save,restore} to mark the whole scope which cannot
// perform any IO with a short explanation why. All allocation requests
// will inherit GFP_NOIO implicitly.
//
// %GFP_NOFS will use direct reclaim but will not use any filesystem interfaces.
// Please try to avoid using this flag directly and instead use
// memalloc_nofs_{save,restore} to mark the whole scope which cannot/shouldn't
// recurse into the FS layer with a short explanation why. All allocation
// requests will inherit GFP_NOFS implicitly.
//
// %GFP_USER is for userspace allocations that also need to be directly
// accessibly by the kernel or hardware. It is typically used by hardware
// for buffers that are mapped to userspace (e.g. graphics) that hardware
// still must DMA to. cpuset limits are enforced for these allocations.
//
// %GFP_DMA exists for historical reasons and should be avoided where possible.
// The flags indicates that the caller requires that the lowest zone be
// used (%ZONE_DMA or 16M on x86-64). Ideally, this would be removed but
// it would require careful auditing as some users really require it and
// others use the flag to avoid lowmem reserves in %ZONE_DMA and treat the
// lowest zone as a type of emergency reserve.
//
// %GFP_DMA32 is similar to %GFP_DMA except that the caller requires a 32-bit
// address. Note that kmalloc(..., GFP_DMA32) does not return DMA32 memory
// because the DMA32 kmalloc cache array is not implemented.
// (Reason: there is no such user in kernel).
//
// %GFP_HIGHUSER is for userspace allocations that may be mapped to userspace,
// do not need to be directly accessible by the kernel but that cannot
// move once in use. An example may be a hardware allocation that maps
// data directly into userspace but has no addressing limitations.
//
// %GFP_HIGHUSER_MOVABLE is for userspace allocations that the kernel does not
// need direct access to but can use kmap() when access is required. They
// are expected to be movable via page reclaim or page migration. Typically,
// pages on the LRU would also be allocated with %GFP_HIGHUSER_MOVABLE.
//
// %GFP_TRANSHUGE and %GFP_TRANSHUGE_LIGHT are used for THP allocations. They
// are compound allocations that will generally fail quickly if memory is not
// available and will not wake kswapd/kcompactd on failure. The _LIGHT
// version does not attempt reclaim/compaction at all and is by default used
// in page fault path, while the non-light is used by khugepaged.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
// early_idt_handler_array is an array of entry points referenced in the
// early IDT.  For simplicity, it's a real array with one entry point
// every nine bytes.  That leaves room for an optional 'push $0' if the
// vector has no error code (two bytes), a 'push $vector_number' (two
// bytes), and a jump to the common entry code (up to five bytes).
// Load a segment. Fall back on loading the zero segment if something goes
// wrong.  This variant assumes that loading zero fully clears the segment.
// This is always the case on Intel CPUs and, even on 64-bit AMD CPUs, any
// failure to fully clear the cached descriptor is only observable for
// FS and GS.
// Save a segment register away:
// __CPUID_LEAF() - Define a CPUID output and parse info entry
// @_name:	Struct type name of the CPUID leaf/subleaf (e.g. 'leaf_0x7_0'). Such
// 		types are defined at <cpuid/leaf_types.h> and follow the leaf_0xM_N
// 		format, where 0xM is the leaf and N is the subleaf.
// @_count:	Number of storage entries to allocate for this leaf/subleaf.
//
// For a given leaf/subleaf, define an array of CPUID storage entries and an associated
// query info structure.
//
// Use an array of storage entries to accommodate CPUID leaves with multiple subleaves
// having the same output format.  This is common for hierarchical enumeration; e.g.,
// CPUID(0x4), CPUID(0x12), and CPUID(0x8000001d).
// CPUID_LEAF() - Define a 'struct cpuid_leaves' storage entry
// @_leaf:	Leaf number, in compile-time 0xN format
// @_subleaf:	Subleaf number, in compile-time decimal format
//
// Convenience wrapper around __CPUID_LEAF().
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Same as rb_first(), but O(1)
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// read_seqcount_begin() - begin a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount() - read the raw seqcount_t counter value
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// raw_read_seqcount opens a read critical section of the given
// seqcount_t, without any lockdep checking, and without checking or
// masking the sequence counter LSB. Calling code is responsible for
// handling that.
//
// Return: count to be passed to read_seqcount_retry()
// raw_seqcount_try_begin() - begin a seqcount_t read critical section
//                            w/o lockdep and w/o counter stabilization
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count to be passed to read_seqcount_retry()
//
// Similar to raw_seqcount_begin(), except it enables eliding the critical
// section entirely if odd, instead of doing the speculation knowing it will
// fail.
//
// Useful when counter stabilization is more or less equivalent to taking
// the lock and there is a slowpath that does that.
//
// If true, start will be set to the (even) sequence count read.
//
// Return: true when a read critical section is started.
// read_seqcount_retry() - end a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count, from read_seqcount_begin()
//
// read_seqcount_retry closes the read critical section of given
// seqcount_t.  If the critical section was invalid, it must be ignored
// (and typically retried).
//
// Return: true if a read section retry is required, else false
// seqlock_init() - dynamic initializer for seqlock_t
// @sl: Pointer to the seqlock_t instance
// read_seqlock_excl_irqsave() - start a non-interruptible seqlock_t
// 				 locking reader section
// @lock:  Pointer to seqlock_t
// @flags: Stack-allocated storage for saving caller's local interrupt
//         state, to be passed to read_sequnlock_excl_irqrestore().
//
// _irqsave variant of read_seqlock_excl(). Use this only if the seqlock_t
// write side section, *or other read sections*, can be invoked from a
// hardirq context.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
// Page flags policies wrt compound pages
//
// PF_POISONED_CHECK
//     check if this struct page poisoned/uninitialized
//
// PF_ANY:
//     the page flag is relevant for small, head and tail pages.
//
// PF_HEAD:
//     for compound page all operations related to the page flag applied to
//     head page.
//
// PF_NO_TAIL:
//     modifications of the page flag must be done on small or head pages,
//     checks can be done on tail pages too.
//
// PF_NO_COMPOUND:
//     the page flag is not relevant for compound pages.
//
// PF_SECOND:
//     the page flag is stored in the first tail page.
// Which page is the flag stored in
// Macros to create function definitions for page flags
// On an anonymous folio mapped into a user virtual memory area,
// folio->mapping points to its anon_vma, not to a struct address_space;
// with the FOLIO_MAPPING_ANON bit set to distinguish it.  See rmap.h.
//
// On an anonymous folio in a VM_MERGEABLE area, if CONFIG_KSM is enabled,
// the FOLIO_MAPPING_ANON_KSM bit may be set along with the FOLIO_MAPPING_ANON
// bit; and then folio->mapping points, not to an anon_vma, but to a private
// structure which KSM associates with that merged folio.  See ksm.h.
//
// Please note that, confusingly, "folio_mapping" refers to the inode
// address_space which maps the folio from disk; whereas "folio_mapped"
// refers to user virtual address space into which the folio is mapped.
//
// For slab pages, since slab reuses the bits in struct page to store its
// internal states, the folio->mapping does not exist as such, nor do
// these flags below.  So in order to avoid testing non-existent bits,
// please make sure that folio_test_slab(folio) actually evaluates to
// false before calling the following functions (e.g., folio_test_anon).
// See mm/slab.h.
// local_lock - Acquire a per CPU local lock
// @lock:	The lock variable
// local_lock_irq - Acquire a per CPU local lock and disable interrupts
// @lock:	The lock variable
// local_lock_irqsave - Acquire a per CPU local lock, save and disable
// 			 interrupts
// @lock:	The lock variable
// @flags:	Storage for interrupt flags
// local_unlock - Release a per CPU local lock
// @lock:	The lock variable
// local_unlock_irq - Release a per CPU local lock and enable interrupts
// @lock:	The lock variable
// local_unlock_irqrestore - Release a per CPU local lock and restore
// 			      interrupt flags
// @lock:	The lock variable
// @flags:      Interrupt flags to restore
// local_trylock_init - Runtime initialize a lock instance
// @lock:	The lock variable
// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// Implementation specific max size
// list_entry_rcu - get the struct for this entry
// @ptr:        the &struct list_head pointer.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// list_next_or_null_rcu - get the next element from a list
// @head:	the head for the list.
// @ptr:        the list head to take the next element from.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// Note that if the ptr is at the end of the list, NULL is returned.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// return the first or the next element in an RCU protected hlist
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// Please ignore the name of this function.  It should be called
// switch_to_kernel_thread().
//
// enter_lazy_tlb() is a hint from the scheduler that we are entering a
// kernel thread or other context without an mm.  Acceptable implementations
// include doing nothing whatsoever, switching to init_mm, or various clever
// lazy tricks to try to minimize TLB flushes.
//
// The scheduler reserves the right to call enter_lazy_tlb() several times
// in a row.  It will notify us that we're going back to a real mm by
// calling switch_mm_irqs_off().
// Keep stack-allocated flush_tlb_info cacheline aligned, but cap the
// alignment to avoid excessive stack usage on large-cacheline systems.
// This is used for both get_user() and __get_user() to expand to
// the proper special function call that has odd calling conventions
// due to returning both a value and an error, and that depends on
// the size of the pointer passed in.
//
// Careful: we have to cast the result to the type of the pointer
// for sign reasons.
//
// The use of _ASM_DX as the register specifier is a bit of a
// simplification, as gcc only cares about it as the starting point
// and not size: for a 64-bit value it will use %ecx:%edx on 32 bits
// (%ecx being the next register in gcc's x86 register sequence), and
// %rdx on 64 bits.
//
// Clang/LLVM cares about the size of the register, but still wants
// the base register for something that ends up being a pair.
// get_user - Get a simple variable from user space.
// @x:   Variable to store result.
// @ptr: Source address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple variable from user space to kernel
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and the result of
// dereferencing @ptr must be assignable to @x without a cast.
//
// Return: zero on success, or -EFAULT on error.
// On error, the variable @x is set to zero.
// ptr must be evaluated and assigned to the temporary __ptr_pu before
// the assignment of x to __val_pu, to avoid any function calls
// involved in the ptr expression (possibly implicitly generated due
// to KASAN) from clobbering %ax.
// eval x once
// eval ptr once
// put_user - Write a simple value into user space.
// @x:   Value to copy to user space.
// @ptr: Destination address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple value from kernel space to user
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and @x must be assignable
// to the result of dereferencing @ptr.
//
// Return: zero on success, or -EFAULT on error.
// The bottom two bits of the slot determine how the remaining bits in the
// slot are interpreted:
//
// 00 - data pointer
// 10 - internal entry
// x1 - value entry
//
// The internal entry may be a pointer to the next level in the tree, a
// sibling entry, or an indicator that the entry in this slot has been moved
// to another location in the tree and the lookup should be restarted.  While
// NULL fits the 'data pointer' pattern, it means that there is no entry in
// the tree for this index (no matter what level of the tree it is found at).
// This means that storing a NULL entry in the tree is the same as deleting
// the entry from the tree.
// direct disk access hint
// Reserved kernel ranges [-100], [-10000, -40000].
// Special value for dirfd used to
// indicate openat should use the
// current working directory.
// Reserved for per-syscall flags	0xff.
// Do not follow symbolic
// links.
// Allow empty relative
// pathname to operate on dirfd
// directly.
// The 10 bits between the priority class and the priority level are used to
// optionally define I/O hints for any combination of I/O priority class and
// level. Depending on the kernel configuration, I/O scheduler being used and
// the target I/O device being used, hints can influence how I/Os are processed
// without affecting the I/O scheduling ordering defined by the I/O priority
// class and level.
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// Set the IDR flag and the IDR_FREE tag
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
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
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__35 {
() => {
((core::mem::size_of::<[__SlateRecord_control_c_1931; 3]>() as u64) / (core::mem::size_of::<__SlateRecord_control_c_1931>() as u64)).wrapping_add(__must_be_array__16!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__37 {
() => {
((core::mem::size_of::<[__SlateRecord_control_c_1931; 3]>() as u64) / (core::mem::size_of::<__SlateRecord_control_c_1931>() as u64)).wrapping_add(__must_be_array__39!() as u64)
};
}

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_WORD {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED__2 {
() => {
__or!()
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// IS_ERR_VALUE - Detect an error pointer.
/// @x: The pointer to check.
///
/// Like IS_ERR(), but does not generate a compiler warning if result is unused.
macro_rules! IS_ERR_VALUE {
($x:expr) => {
unlikely!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! MINOR {
($dev:expr) => {
($dev) & 1048575u32
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! SNDRV_PROTOCOL_VERSION {
($major:expr, $minor:expr, $subminor:expr) => {
($major) << 16i32 | ($minor) << 8i32 | $subminor
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__2 {
() => {
16045481047390945280u64
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut list_head, $val) };
if true {
break;
}
}
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

macro_rules! ____or {
() => {
1i32
};
}

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire {
() => {
__acquire_ctx_lock!();
};
}

macro_rules! __acquire_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __assume_ctx_lock {
($var:expr) => {
loop {
$var;
if true {
break;
}
}
};
}

macro_rules! __chk_user_ptr {
() => {
};
}

/// __flex_counter() - Get pointer to counter member for the given
///                    flexible array, if it was annotated with __counted_by()
/// @FAM: Pointer to flexible array member of an addressable struct instance
///
/// For example, with:
///
/// 	struct foo {
/// 		int counter;
/// 		short array[] __counted_by(counter);
/// 	} *p;
///
/// __flex_counter(p->array) will resolve to &p->counter.
///
/// Note that Clang may not allow this to be assigned to a separate
/// variable; it must be used directly.
///
/// If p->array is unannotated, this returns (void *)NULL.
macro_rules! __flex_counter__3 {
($__obj_ptr:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__obj_ptr).count) }
};
}

macro_rules! __get_and_null {
($p:expr) => {
{
let mut __v: *mut u8;
let __v2: *mut *mut u8 = core::ptr::addr_of_mut!($p);
let __v3: *mut u8 = unsafe { *__v2 };
unsafe {
*__v2 = core::ptr::null_mut::<u8>();
}
__v = __v3;
__v
}
};
}

macro_rules! __get_and_null__3 {
($p:expr) => {
{
let mut __v: *mut u32;
let __v2: *mut *mut u32 = core::ptr::addr_of_mut!($p);
let __v3: *mut u32 = unsafe { *__v2 };
unsafe {
*__v2 = core::ptr::null_mut::<u32>();
}
__v = __v3;
__v
}
};
}

/// Getting something that works in C and CPP for an arg that may or may
/// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
/// we match on the placeholder define, insert the "0," for arg1 and generate
/// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
/// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
/// the last step cherry picks the 2nd arg, we get a zero.
macro_rules! __is_defined {
() => {
____is_defined!()
};
}

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__16 {
() => {
core::mem::size_of::<__SlateRecord_control_c_1940>() as i32
};
}

macro_rules! __must_be_array__39 {
() => {
core::mem::size_of::<__SlateRecord_control_c_1944>() as i32
};
}

macro_rules! __must_be_array__48 {
() => {
core::mem::size_of::<__SlateRecord_control_c_239_4>() as i32
};
}

macro_rules! __must_be_array__54 {
() => {
core::mem::size_of::<__SlateRecord_control_c_1566>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr {
() => {
core::mem::size_of::<__SlateRecord_control_c_1566_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__4 {
() => {
core::mem::size_of::<__SlateRecord_control_c_2580_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__25 {
() => {
core::mem::size_of::<__SlateRecord_control_c_894>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__27 {
() => {
core::mem::size_of::<__SlateRecord_control_c_891_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__29 {
() => {
core::mem::size_of::<__SlateRecord_control_c_893>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__36 {
() => {
core::mem::size_of::<__SlateRecord_control_c_299_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__40 {
() => {
core::mem::size_of::<__SlateRecord_control_c_1566_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__41 {
() => {
core::mem::size_of::<__SlateRecord_control_c_784_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__42 {
() => {
core::mem::size_of::<__SlateRecord_control_c_2580>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__54 {
() => {
core::mem::size_of::<__SlateRecord_control_c_784>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__64 {
() => {
core::mem::size_of::<__SlateRecord_control_c_890_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__66 {
() => {
core::mem::size_of::<__SlateRecord_control_c_889_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__74 {
() => {
core::mem::size_of::<__SlateRecord_control_c_891>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__81 {
() => {
core::mem::size_of::<__SlateRecord_control_c_892>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__87 {
() => {
core::mem::size_of::<__SlateRecord_control_c_892_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__88 {
() => {
core::mem::size_of::<__SlateRecord_control_c_893_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__94 {
() => {
core::mem::size_of::<__SlateRecord_control_c_299>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__97 {
() => {
core::mem::size_of::<__SlateRecord_control_c_894_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__98 {
() => {
core::mem::size_of::<__SlateRecord_control_c_889>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__103 {
() => {
core::mem::size_of::<__SlateRecord_control_c_890>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
() => {
__release_ctx_lock!();
};
}

macro_rules! __release_ctx_lock {
() => {
while false {
}
};
}

/// __set_flex_counter() - Set the counter associated with the given flexible
///                        array member that has been annoated by __counted_by().
/// @FAM: Instance of flexible array member within a given struct.
/// @COUNT: Value to store to the __counted_by annotated @FAM_PTR's counter.
///
/// This is a no-op if no annotation exists. Count needs to be checked with
/// overflows_flex_counter_type() before using this function.
macro_rules! __set_flex_counter {
($COUNT:expr, $__obj_ptr:expr) => {
let mut __v: u32;
let __v2: u32 = ($COUNT) as u32;
unsafe {
*__flex_counter__3!($__obj_ptr) = __v2;
}
__v = __v2;
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

/// check_add_overflow() - Calculate addition with overflow checking
/// @a: first addend
/// @b: second addend
/// @d: pointer to store sum
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted addition, regardless of whether
/// wrap-around occurred.
macro_rules! check_add_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_add(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_add(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_mul(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__20 {
($ptr:expr) => {
{
let mut __v: *mut snd_kctl_ioctl;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__35!() as isize)) }) as *mut ()) as *mut snd_kctl_ioctl;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__23 {
($ptr:expr) => {
{
let mut __v: *mut snd_ctl_file;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__19!() as isize)) }) as *mut ()) as *mut snd_ctl_file;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__28 {
($ptr:expr) => {
{
let mut __v: *mut snd_kctl_event;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__52!() as isize)) }) as *mut ()) as *mut snd_kctl_event;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__43 {
($ptr:expr) => {
{
let mut __v: *mut snd_kcontrol;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut snd_kcontrol;
__v
}
};
}

macro_rules! debug_normal_state_change {
() => {
while false {
}
};
}

/// flex_array_size() - Calculate size of a flexible array member
///                     within an enclosing structure.
/// @p: Pointer to the structure.
/// @member: Name of the flexible array member.
/// @count: Number of elements in the array.
///
/// Calculates size of a flexible array of @count number of @member
/// elements, at the end of structure @p.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! flex_array_size {
($count:expr) => {
size_mul($count, (core::mem::size_of::<snd_kcontrol_volatile>() as u64).wrapping_add(__must_be_array__48!() as u64))
};
}

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user {
($to:expr) => {
let mut __v: u32;
let mut __tmp: u64 = ($to) as u64;
kmsan_unpoison_memory(core::ptr::addr_of_mut!(__tmp) as *const (), core::mem::size_of::<u64>() as u64);
let __v2: u32 = __tmp as u32;
$to = __v2;
__v = __v2;
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<i32>() as u64, 0u64);
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user__2 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u32>() as u64, 0u64);
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__9 {
($ptr:expr) => {
container_of__20!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__19 {
($ptr:expr) => {
container_of__23!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__22 {
($ptr:expr) => {
container_of__28!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__23 {
($ptr:expr) => {
container_of__43!($ptr)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__7 {
($ptr:expr) => {
list_entry__23!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__11 {
($ptr:expr) => {
list_entry__19!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__12 {
($ptr:expr) => {
list_entry__9!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__26 {
($ptr:expr) => {
list_entry__22!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__8 {
($pos:expr, $member:ident) => {
list_entry__23!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__19 {
($pos:expr, $member:ident) => {
list_entry__22!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__13 {
($pos:expr, $member:ident) => {
list_entry__9!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__14 {
($pos:expr, $member:ident) => {
list_entry__19!(unsafe { (*$pos).$member.next })
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_assert_held_write {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! might_sleep {
() => {
loop {
might_resched();
if true {
break;
}
}
};
}

macro_rules! might_sleep_if {
($cond:expr) => {
loop {
if $cond {
might_sleep!();
}
if true {
break;
}
}
};
}

macro_rules! no_free_ptr__2 {
($p:expr) => {
__must_check_fn(__get_and_null__3!($p) as *const ()) as *mut u32
};
}

macro_rules! no_free_ptr__3 {
($p:expr) => {
__must_check_fn(__get_and_null!($p) as *const ()) as *mut u8
};
}

macro_rules! offsetof__19 {
() => {
core::mem::offset_of!(snd_ctl_file, list) as u64
};
}

macro_rules! offsetof__35 {
() => {
core::mem::offset_of!(snd_kctl_ioctl, list) as u64
};
}

macro_rules! offsetof__5 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, list) as u64
};
}

macro_rules! offsetof__52 {
() => {
core::mem::offset_of!(snd_kctl_event, list) as u64
};
}

macro_rules! offsetof__6 {
() => {
0u64
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_spin_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_spin_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! read_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_read_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

macro_rules! read_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_read_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! snd_BUG_ON__5 {
($condition:expr) => {
let mut __v: i64;
let __ret_warn_on: i32 = (($condition) != 0) as i32;
__v = (__ret_warn_on != 0) as i64;
};
}

macro_rules! snd_BUG_ON {
($condition:expr) => {
{
let mut __v: i64;
let __v2: i32 = !(!$condition) as i32;
__v = (__v2 != 0) as i64;
__v
}
};
}

macro_rules! snd_BUG {
() => {
while false {
}
};
}

macro_rules! snd_ctl_skip_validation {
() => {
1i32
};
}

macro_rules! snd_kcontrol {
($n:expr) => {
{
let mut __v: *mut snd_kcontrol;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut snd_kcontrol;
__v
}
};
}

macro_rules! snd_kcontrol_chip {
($kcontrol:expr) => {
unsafe { (*$kcontrol).private_data }
};
}

macro_rules! snd_kctl_event {
($n:expr) => {
{
let mut __v: *mut snd_kctl_event;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!(snd_kctl_event) as isize)) }) as *mut ()) as *mut snd_kctl_event;
__v
}
};
}

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release!();
__acquire!();
if true {
break;
}
}
};
}

macro_rules! static_branch_unlikely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

/// Useful shorthand for "is this condition known at compile-time?"
///
/// Note that the condition may involve non-constant values,
/// but the compiler may know enough about the details of the
/// values to determine that the condition is statically true.
macro_rules! statically_true {
($x:expr) => {
(unsafe { __slate_intrinsic_1cfe0293d0a2af8d($x) }) && $x
};
}

/// struct_size() - Calculate size of structure with trailing flexible array.
/// @p: Pointer to the structure.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure of @p followed by an
/// array of @count number of @member elements.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size {
($count:expr) => {
size_add(144u64, flex_array_size!($count))
};
}

/// struct_size_t() - Calculate size of structure with trailing flexible array
/// @type: structure type name.
/// @member: Name of the array member.
/// @count: Number of elements in the array.
///
/// Calculates size of memory needed for structure @type followed by an
/// array of @count number of @member elements. Prefer using struct_size()
/// when possible instead, to keep calculations associated with a specific
/// instance variable of type @type.
///
/// Return: number of bytes needed or SIZE_MAX on overflow.
macro_rules! struct_size_t__2 {
($count:expr) => {
struct_size!($count)
};
}

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! trace_set_current_state {
($state_value:expr) => {
loop {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck {
($type:ty) => {
let mut __v: i32;
let mut __dummy: $type = 0;
let mut __dummy2: $type = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v = 1;
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__4 {
($type:ty) => {
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
};
}

/// u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
/// @x: The u64 value from user space, usually via IOCTL
///
/// u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
/// __user * correctly. Using this lets us get rid of all the tiresome casts.
macro_rules! u64_to_user_ptr {
($x:expr) => {
{
let mut __v: *mut ();
typecheck__4!(u64);
__v = (($x) as u64) as *mut ();
__v
}
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! wake_up {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

macro_rules! write_lock_irq {
($lock:expr) => {
unsafe { _raw_write_lock_irq($lock) };
};
}

macro_rules! write_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_write_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

macro_rules! write_unlock_irq {
($lock:expr) => {
unsafe { _raw_write_unlock_irq($lock) };
};
}

macro_rules! write_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_write_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! xa_lock {
($xa:expr) => {
spin_lock(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

macro_rules! xa_unlock {
($xa:expr) => {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*$xa).xa_lock) });
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut __invalid_size_argument_for_IOC: u32;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut __tracepoint_sched_set_state_tp: tracepoint;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __bad_size_call_parameter();
    fn __xchg_wrong_size();
    fn __xadd_wrong_size();
    fn memdup_user(arg0: *const (), arg1: u64) -> *mut ();
    fn vmemdup_user(arg0: *const (), arg1: u64) -> *mut ();
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn strlen(arg0: *const u8) -> u64;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn strnlen(arg0: *const u8, arg1: u64) -> u64;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_21();
    fn __compiletime_assert_22();
    fn __compiletime_assert_62();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn _raw_write_lock_irq(arg0: *mut rwlock);
    fn _raw_read_lock_irqsave(arg0: *mut rwlock) -> u64;
    fn _raw_write_lock_irqsave(arg0: *mut rwlock) -> u64;
    fn _raw_write_unlock_irq(arg0: *mut rwlock);
    fn _raw_read_unlock_irqrestore(arg0: *mut rwlock, arg1: u64);
    fn _raw_write_unlock_irqrestore(arg0: *mut rwlock, arg1: u64);
    fn schedule();
    fn __trace_set_current_state(arg0: i32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn default_wake_function(arg0: *mut wait_queue_entry, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn add_wait_queue(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn remove_wait_queue(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn down_read(arg0: *mut rw_semaphore);
    fn down_write(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn downgrade_write(arg0: *mut rw_semaphore);
    fn __request_module(arg0: bool, arg1: *const u8, ...) -> i32;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn xa_load(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn xa_erase(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn xa_store_range(arg0: *mut xarray, arg1: u64, arg2: u64, arg3: *mut (), arg4: u32) -> *mut ();
    fn xa_destroy(arg0: *mut xarray);
    fn __xa_insert(arg0: *mut xarray, arg1: u64, arg2: *mut (), arg3: u32) -> i32;
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn put_pid(arg0: *mut pid);
    fn pid_vnr(arg0: *mut pid) -> i32;
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn kvfree(arg0: *const ());
    fn stream_open(arg0: *mut inode, arg1: *mut file) -> i32;
    fn try_module_get(arg0: *mut module) -> bool;
    fn module_put(arg0: *mut module);
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn dev_set_name(arg0: *mut device, arg1: *const u8, ...) -> i32;
    fn put_device(arg0: *mut device);
    fn snd_refcount_put(arg0: *mut snd_refcount);
    fn snd_power_ref_and_wait(arg0: *mut snd_card) -> i32;
    fn snd_device_alloc(arg0: *mut *mut device, arg1: *mut snd_card) -> i32;
    fn snd_register_device(arg0: i32, arg1: *mut snd_card, arg2: i32, arg3: *const file_operations, arg4: *mut (), arg5: *mut device) -> i32;
    fn snd_unregister_device(arg0: *mut device) -> i32;
    fn snd_lookup_minor_data(arg0: u32, arg1: i32) -> *mut ();
    fn snd_card_file_add(arg0: *mut snd_card, arg1: *mut file) -> i32;
    fn snd_card_file_remove(arg0: *mut snd_card, arg1: *mut file) -> i32;
    fn snd_card_ref(arg0: i32) -> *mut snd_card;
    fn snd_device_new(arg0: *mut snd_card, arg1: u32, arg2: *mut (), arg3: *const snd_device_ops) -> i32;
    fn snd_fasync_helper(arg0: i32, arg1: *mut file, arg2: i32, arg3: *mut *mut snd_fasync) -> i32;
    fn snd_kill_fasync(arg0: *mut snd_fasync, arg1: i32, arg2: i32);
    fn snd_fasync_free(arg0: *mut snd_fasync);
    fn __compiletime_assert_503();
    fn __compiletime_assert_504();
    fn __compiletime_assert_505();
    fn __compiletime_assert_506();
    fn __compiletime_assert_570();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

// generic data direction definitions
// The type of an index into the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_197 {
    counter: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_head {
    first: *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_node {
    next: *mut hlist_node,
    pprev: *mut *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcuwait {
    task: *mut task_struct,
}


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
//
// @ptr: address of access
// @size: size of access
// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
//
// Assert that there are no concurrent writes to @var; other readers are
// allowed. This assertion can be used to specify properties of concurrent code,
// where violation cannot be detected as a normal data race.
//
// For example, if we only have a single writer, but multiple concurrent
// readers, to avoid data races, all these accesses must be marked; even
// concurrent marked writes racing with the single writer are bugs.
// Unfortunately, due to being marked, they are no longer data races. For cases
// like these, we can use the macro as follows:
//
// .. code-block:: c
//
// 	void writer(void) {
// 		spin_lock(&update_foo_lock);
// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
// 		WRITE_ONCE(shared_foo, ...);
// 		spin_unlock(&update_foo_lock);
// 	}
// 	void reader(void) {
// 		// update_foo_lock does not need to be held!
// 		... = READ_ONCE(shared_foo);
// 	}
//
// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
// checking if a clear scope where no concurrent writes are expected exists.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_ACCESS - assert no concurrent accesses to @var
//
// Assert that there are no concurrent accesses to @var (no readers nor
// writers). This assertion can be used to specify properties of concurrent
// code, where violation cannot be detected as a normal data race.
//
// For example, where exclusive access is expected after determining no other
// users of an object are left, but the object is not actually freed. We can
// check that this property actually holds as follows:
//
// .. code-block:: c
//
// 	if (refcount_dec_and_test(&obj->refcnt)) {
// 		ASSERT_EXCLUSIVE_ACCESS(*obj);
// 		do_some_cleanup(obj);
// 		release_for_reuse(obj);
// 	}
//
// Note:
//
// 1. ASSERT_EXCLUSIVE_ACCESS_SCOPED(), if applicable, performs more thorough
//    checking if a clear scope where no concurrent accesses are expected exists.
//
// 2. For cases where the object is freed, `KASAN <kasan.html>`_ is a better
//    fit to detect use-after-free bugs.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
//
// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
//
// Assert that there are no concurrent writes to a subset of bits in @var;
// concurrent readers are permitted. This assertion captures more detailed
// bit-level properties, compared to the other (word granularity) assertions.
// Only the bits set in @mask are checked for concurrent modifications, while
// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
// are ignored.
//
// Use this for variables, where some bits must not be modified concurrently,
// yet other bits are expected to be modified concurrently.
//
// For example, variables where, after initialization, some bits are read-only,
// but other bits may still be modified concurrently. A reader may wish to
// assert that this is true as follows:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
// to access the masked bits only, and KCSAN optimistically assumes it is
// therefore safe, even in the presence of data races, and marking it with
// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
// it may still be advisable to do so, since we cannot reason about all compiler
// optimizations when it comes to bit manipulations (on the reader and writer
// side). If you are sure nothing can go wrong, we can write the above simply
// as:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Another example, where this may be used, is when certain bits of @var may
// only be modified when holding the appropriate lock, but other bits may still
// be modified concurrently. Writers, where other bits may change concurrently,
// could use the assertion as follows:
//
// .. code-block:: c
//
// 	spin_lock(&foo_lock);
// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
// 	old_flags = flags;
// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
// 	spin_unlock(&foo_lock);
//
// @var: variable to assert on
// @mask: only check for modifications to bits set in @mask
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
}


// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

#[inline(always)]
fn size_add(addend1: u64, addend2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_add_overflow!(addend1, addend2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qspinlock_types_h_44 {
    val: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qspinlock_types_h_44_2,
    __slate_anon_2: __SlateRecord_qspinlock_types_h_44_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_2 {
    locked: u8,
    pending: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_3 {
    locked_pending: u16,
    tail: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qrwlock_types_h_27 {
    cnts: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qrwlock_types_h_27_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qrwlock_types_h_27_2 {
    wlocked: u8,
    __lstate: [u8; 3],
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
#[repr(C)]
#[derive(Clone, Copy)]
struct qspinlock {
    __slate_anon_0: __SlateRecord_qspinlock_types_h_44,
}


// Bitfields in the atomic value:
//
// When NR_CPUS < 16K
//  0- 7: locked byte
//  8-15: pending byte
// 16-17: tail index
// 18-31: tail cpu (+1)
//
// When NR_CPUS >= 16K
//  0- 7: locked byte
//     8: pending
//  9-10: tail index
// 11-31: tail cpu (+1)
#[repr(C)]
#[derive(Clone, Copy)]
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lock_class_key {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ratelimit_state {
    lock: raw_spinlock,
    interval: i32,
    burst: i32,
    rs_n_left: __SlateRecord_types_h_190,
    missed: __SlateRecord_types_h_190,
    flags: u32,
    begin: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[inline(always)]
fn variable_test_bit(mut nr: i64, mut addr: *const u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut u64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
}

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

fn kmsan_copy_to_user(to: *mut (), from: *const (), to_copy: u64, left: u64) {
}

#[inline(always)]
fn instrument_atomic_check_alignment(v: *const (), size: u64) {
if IS_ENABLED!() != 0 {
let mut mask: u32 = size.wrapping_sub(1) as u32;
if IS_ENABLED!() != 0 {
mask = ((mask as u64) & (core::mem::size_of::<__SlateRecord_instrumented_h_66>() as u64).wrapping_sub(1)) as u32;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((v as u64) & (mask as u64) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("2: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 2b - ., 3; .popsection\n# {0}", const 2 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_instrument_atomic_check_alignment_3, sym __slate_asm_string_instrument_atomic_check_alignment_4, const 67 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 4; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_4: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

fn might_fault() {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_29 {
    futex: __SlateRecord_restart_block_h_31,
    nanosleep: __SlateRecord_restart_block_h_40,
    poll: __SlateRecord_restart_block_h_50,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_31 {
    uaddr: *mut u32,
    val: u32,
    flags: u32,
    bitset: u32,
    time: i64,
    uaddr2: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_40 {
    clockid: i32,
    r#type: u32,
    __slate_anon_2: __SlateRecord_restart_block_h_43,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_43 {
    rmtp: *mut __kernel_timespec,
    compat_rmtp: *mut old_timespec32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_50 {
    ufds: *mut pollfd,
    nfds: i32,
    has_timeout: i32,
    end_time: timespec64,
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn __must_check_fn(val: *const ()) -> *const () {
val
}

// Helper macro for scoped_guard().
//
// Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
// compiler would be sure that for the unconditional locks the body of the
// loop (caller-provided code glued to the else clause) could not be skipped.
// It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
// hard to deduce (even if could be proven true for unconditional locks).
// To support Context Analysis, we need to allow the compiler to see the
// acquisition and release of the context lock. However, the "cleanup" helpers
// wrap the lock in a struct passed through separate helper functions, which
// hides the lock alias from the compiler (no inter-procedural analysis).
//
// To make it work, we introduce an explicit alias to the context lock instance
// that is "cleaned" up with a separate cleanup helper. This helper is a dummy
// function that does nothing at runtime, but has the "_unlock" attribute to
// tell the compiler what happens at the end of the scope.
//
// To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
// to redefine the constructor, which then also creates the alias variable with
// the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
// used.
//
// Example usage:
//
//   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//
// Note: To support the for-loop based scoped helpers, the auxiliary variable
// must be a pointer to the "class" type because it is defined in the same
// statement as the guard variable. However, we initialize it with the lock
// pointer (despite the type mismatch, the compiler's alias analysis still works
// as expected). The "_unlock" attribute receives a pointer to the auxiliary
// variable (a double pointer to the class type), and must be cast and
// dereferenced appropriately.
#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


#[inline(always)]
fn arch_atomic_fetch_add(mut __ret: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("lock xaddl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


fn memset32(mut s: *mut u32, mut v: u32, mut n: u64) -> *mut () {
let s0: *mut u32 = s;
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("rep stosl", inlateout("di") s as u64 => __t0, inlateout("cx") n as u64 => __t1, in("ax") v as u32, options(nostack, att_syntax));
s = unsafe { core::mem::transmute::<u64, *mut u32>(__t0 as u64) };
n = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
s0 as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_17() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).prev, list);
if true {
break;
}
}
}

fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[inline(always)]
fn __list_add(new: *mut list_head, prev: *mut list_head, next: *mut list_head) {
if !__list_add_valid(new, prev, next) {
return;
}
unsafe {
(*next).prev = new;
}
unsafe {
(*new).next = next;
}
unsafe {
(*new).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_18() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, new);
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_21() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, next);
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_22() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
}


fn div64_u64_rem(dividend: u64, divisor: u64, remainder: &mut u64) -> u64 {
unsafe {
*(remainder as *mut u64) = dividend % divisor;
}
dividend / divisor
}

// DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
// @ll: unsigned 64bit dividend
// @d: unsigned 32bit divisor
//
// Divide unsigned 64bit dividend by unsigned 32bit divisor
// and round up.
//
// Return: dividend / divisor rounded up
#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_62() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = core::ptr::null_mut::<task_struct>();
return __v0;
}
let mut __v2: *mut task_struct;
let mut pscr_ret__: *mut task_struct = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<*mut task_struct>() }) as u64 {
1 => {
let mut __v4: *mut task_struct;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v4 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v4;
}
2 => {
let mut __v6: *mut task_struct;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v6 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v6;
}
4 => {
let mut __v7: *mut task_struct;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v7 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v7;
}
8 => {
let mut __v8: *mut task_struct;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v8 = pfo_val__ as *mut task_struct;
pscr_ret__ = __v8;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v2 = pscr_ret__;
__v2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_22 {
    pmd: u64,
}


// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


// was written to (raised by CPU)
// on 4KB pages
// On 2MB or 1GB pages
// Protection Keys, bit 1/4
// Saved Dirty bit (leaf)
// The hardware requires shadow stack to be Write=0,Dirty=1. However,
// there are valid cases where the kernel might create read-only PTEs that
// are dirty (e.g., fork(), mprotect(), userfaultfd, soft-dirty tracking). In
// this case, the _PAGE_SAVED_DIRTY bit is used instead of the HW-dirty bit,
// to avoid creating a wrong "shadow stack" PTEs. Such PTEs have
// (Write=0,SavedDirty=1,Dirty=0) set.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
#[repr(C)]
#[derive(Clone, Copy)]
struct pgprot {
    pgprot: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits144U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits145U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_struct {
    tls_array: [desc_struct; 3],
    sp: u64,
    es: u16,
    ds: u16,
    fsindex: u16,
    gsindex: u16,
    fsbase: u64,
    gsbase: u64,
    ptrace_bps: [*mut perf_event; 4],
    virtual_dr6: u64,
    ptrace_dr7: u64,
    cr2: u64,
    trap_nr: u64,
    error_code: u64,
    io_bitmap: *mut io_bitmap,
    iopl_emul: u64,
    __slate_bits_0: __slate_bits::__SlateBits146U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_39 {
    _pid: i32,
    _uid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_45 {
    _tid: i32,
    _overrun: i32,
    _sigval: sigval,
    _sys_private: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_53 {
    _pid: i32,
    _uid: u32,
    _sigval: sigval,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_60 {
    _pid: i32,
    _uid: u32,
    _status: i32,
    _utime: i64,
    _stime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_69 {
    _addr: *mut (),
    __slate_anon_1: __SlateRecord_siginfo_h_74,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_siginfo_h_74 {
    _trapno: i32,
    _addr_lsb: i16,
    _addr_bnd: __SlateRecord_siginfo_h_83,
    _addr_pkey: __SlateRecord_siginfo_h_89,
    _perf: __SlateRecord_siginfo_h_94,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_83 {
    _dummy_bnd: [u8; 8],
    _lower: *mut (),
    _upper: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_89 {
    _dummy_pkey: [u8; 8],
    _pkey: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_94 {
    _data: u64,
    _type: u32,
    _flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_103 {
    _band: i64,
    _fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_109 {
    _call_addr: *mut (),
    _syscall: i32,
    _arch: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_sched_data {
    robust_list: *mut robust_list_head,
    compat_robust_list: *mut compat_robust_list_head,
    pi_state_list: list_head,
    pi_state_cache: *mut futex_pi_state,
    exit_mutex: mutex,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_phash {
    lock: mutex,
    hash: *mut futex_private_hash,
    hash_new: *mut futex_private_hash,
    batches: u64,
    rcu: callback_head,
    atomic: __SlateRecord_types_h_197,
    r#ref: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer {
    node: timerqueue_linked_node,
    base: *mut hrtimer_clock_base,
    is_queued: bool,
    is_rel: bool,
    is_soft: bool,
    is_hard: bool,
    is_lazy: bool,
    _softexpires: i64,
    function: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_38 {
    all: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_40,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_40 {
    __slate_anon_0: __SlateRecord_rseq_types_h_41,
    has_rseq: u8,
    __pad: u8,
    __slate_anon_3: __SlateRecord_rseq_types_h_52,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_41 {
    events: u32,
    __slate_anon_1: __SlateRecord_rseq_types_h_43,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_43 {
    sched_switch: u8,
    ids_changed: u8,
    user_irq: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_52 {
    error: u16,
    __slate_anon_1: __SlateRecord_rseq_types_h_54,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_54 {
    fatal: u8,
    slowpath: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_74 {
    cpu_cid: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_76,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_76 {
    cpu_id: u32,
    mm_cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __sifields {
    _kill: __SlateRecord_siginfo_h_39,
    _timer: __SlateRecord_siginfo_h_45,
    _rt: __SlateRecord_siginfo_h_53,
    _sigchld: __SlateRecord_siginfo_h_60,
    _sigfault: __SlateRecord_siginfo_h_69,
    _sigpoll: __SlateRecord_siginfo_h_103,
    _sigsys: __SlateRecord_siginfo_h_109,
}


// SIGPOLL (or any other signal without signal specific si_codes) si_codes
// data input available
// i/o error
#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits218U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint {
    name: *const u8,
    key: static_key_false,
    static_call_key: *mut static_call_key,
    static_call_tramp: *mut (),
    iterator: *mut (),
    probestub: *mut (),
    funcs: *mut tracepoint_func,
    ext: *mut tracepoint_ext,
}


// If a tracepoint needs to be called from a header file, it is not
// recommended to call it directly, as tracepoints in header files
// may cause side-effects and bloat the kernel. Instead, use
// tracepoint_enabled() to test if the tracepoint is enabled, then if
// it is, call a wrapper function defined in a C file that will then
// call the tracepoint.
//
// For "trace_foo_bar()", you would need to create a wrapper function
// in a C file to call trace_foo_bar():
//   void do_trace_foo_bar(args) { trace_foo_bar(args); }
// Then in the header file, declare the tracepoint:
//   DECLARE_TRACEPOINT(foo_bar);
// And call your wrapper:
//   static inline void some_inlined_function() {
//            [..]
//            if (tracepoint_enabled(foo_bar))
//                    do_trace_foo_bar(args);
//            [..]
//   }
//
// Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//   but is safe to have in headers, where trace_foo_bar_enabled() is not.
#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_exec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_group {
}


#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

static class_spinlock_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_610 {
    lock: *mut spinlock,
}


#[inline(always)]
fn class_spinlock_destructor(_T: *mut __SlateRecord_spinlock_h_610) {
spin_unlock(unsafe { (*_T).lock });
}

#[inline(always)]
fn class_spinlock_lock_ptr(_T: &mut __SlateRecord_spinlock_h_610) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_610)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_spinlock_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_610 {
let mut _t: __SlateRecord_spinlock_h_610 = __SlateRecord_spinlock_h_610 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(_t);
spin_lock(unsafe { (*_T).lock });
return _t;
}

#[inline(always)]
fn __class_spinlock_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_610) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_642 {
    lock: *mut spinlock,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[inline(always)]
fn class_spinlock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_642) {
spin_unlock_irqrestore(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_spinlock_irqsave_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_642 {
let mut _t: __SlateRecord_spinlock_h_642 = {
let mut __t0: __SlateRecord_spinlock_h_642 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(_t);
spin_lock_irqsave!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[inline(always)]
fn __class_spinlock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_642) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
}


static class_read_lock_irqsave_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_670 {
    lock: *mut rwlock,
    flags: u64,
}


#[inline(always)]
fn class_read_lock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_670) {
read_unlock_irqrestore!(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_read_lock_irqsave_lock_ptr(_T: &mut __SlateRecord_spinlock_h_670) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_670)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[repr(C)]
#[derive(Clone, Copy)]
struct files_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_restriction {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
}


#[inline(always)]
fn class_read_lock_irqsave_constructor(l: *mut rwlock) -> __SlateRecord_spinlock_h_670 {
let mut _t: __SlateRecord_spinlock_h_670 = {
let mut __t0: __SlateRecord_spinlock_h_670 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_670 = core::ptr::addr_of_mut!(_t);
read_lock_irqsave!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[inline(always)]
fn __class_read_lock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_670) {
}

static class_write_lock_irq_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_683 {
    lock: *mut rwlock,
}


#[inline(always)]
fn class_write_lock_irq_destructor(_T: *mut __SlateRecord_spinlock_h_683) {
write_unlock_irq!(unsafe { (*_T).lock });
}

#[inline(always)]
fn class_write_lock_irq_lock_ptr(_T: &mut __SlateRecord_spinlock_h_683) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_683)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_write_lock_irq_constructor(l: *mut rwlock) -> __SlateRecord_spinlock_h_683 {
let mut _t: __SlateRecord_spinlock_h_683 = __SlateRecord_spinlock_h_683 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_683 = core::ptr::addr_of_mut!(_t);
write_lock_irq!(unsafe { (*_T).lock });
return _t;
}

#[inline(always)]
fn __class_write_lock_irq_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_683) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


static class_write_lock_irqsave_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_689 {
    lock: *mut rwlock,
    flags: u64,
}


#[inline(always)]
fn class_write_lock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_689) {
write_unlock_irqrestore!(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_write_lock_irqsave_lock_ptr(_T: &mut __SlateRecord_spinlock_h_689) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_689)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_write_lock_irqsave_constructor(l: *mut rwlock) -> __SlateRecord_spinlock_h_689 {
let mut _t: __SlateRecord_spinlock_h_689 = {
let mut __t0: __SlateRecord_spinlock_h_689 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_689 = core::ptr::addr_of_mut!(_t);
write_lock_irqsave!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn __class_write_lock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_689) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_io_accounting {
    rchar: u64,
    wchar: u64,
    syscr: u64,
    syscw: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_31 {
    insn: [u8; 16],
    ixol: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_38 {
    branch: __SlateRecord_uprobes_h_39,
    defparam: __SlateRecord_uprobes_h_44,
    push: __SlateRecord_uprobes_h_48,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_39 {
    offs: i32,
    ilen: u8,
    opc1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_44 {
    fixups: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_48 {
    reg_offset: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_136 {
    __slate_anon_0: __SlateRecord_uprobes_h_137,
    __slate_anon_1: __SlateRecord_uprobes_h_142,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_137 {
    autask: arch_uprobe_task,
    vaddr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_142 {
    dup_xol_work: callback_head,
    dup_xol_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_mm_cid {
    pcpu: *mut mm_cid_pcpu,
    mode: u32,
    max_cids: u32,
    irq_work: irq_work,
    work: work_struct,
    lock: raw_spinlock,
    mutex: mutex,
    user_list: hlist_head,
    nr_cpus_allowed: u32,
    users: u32,
    pcpu_thrs: u32,
    update_deferred: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_89 {
    __slate_anon_0: __SlateRecord_mm_types_h_90,
    __slate_anon_1: __SlateRecord_mm_types_h_118,
    __slate_anon_2: __SlateRecord_mm_types_h_129,
    __slate_anon_3: __SlateRecord_mm_types_h_132,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_90 {
    __slate_anon_0: __SlateRecord_mm_types_h_96,
    mapping: *mut address_space,
    __slate_anon_2: __SlateRecord_mm_types_h_105,
    private: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_96 {
    lru: list_head,
    buddy_list: list_head,
    pcp_list: list_head,
    pcp_llist: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_105 {
    __folio_index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_118 {
    pp_magic: u64,
    pp: *mut page_pool,
    _pp_mapping_pad: u64,
    dma_addr: u64,
    pp_ref_count: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_129 {
    compound_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_132 {
    _unused_pgmap_compound_info: *mut (),
    zone_device_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_155 {
    page_type: u32,
    _mapcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_406 {
    __slate_anon_0: __SlateRecord_mm_types_h_407,
    page: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_407 {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_410,
    mapping: *mut address_space,
    __slate_anon_3: __SlateRecord_mm_types_h_425,
    __slate_anon_4: __SlateRecord_mm_types_h_429,
    _mapcount: __SlateRecord_types_h_190,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_410 {
    lru: list_head,
    __slate_anon_1: __SlateRecord_mm_types_h_414,
    pgmap: *mut dev_pagemap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_414 {
    __filler: *mut (),
    mlock_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_425 {
    index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_429 {
    private: *mut (),
    swap: __SlateRecord_mm_types_h_288,
    migrate_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_451 {
    __slate_anon_0: __SlateRecord_mm_types_h_452,
    __page_1: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_452 {
    _flags_1: u64,
    _head_1: u64,
    __slate_anon_2: __SlateRecord_mm_types_h_455,
    _mapcount_1: __SlateRecord_types_h_190,
    _refcount_1: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_455 {
    __slate_anon_0: __SlateRecord_mm_types_h_456,
    _usable_1: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_456 {
    _large_mapcount: __SlateRecord_types_h_190,
    _nr_pages_mapped: __SlateRecord_types_h_190,
    _entire_mapcount: __SlateRecord_types_h_190,
    _pincount: __SlateRecord_types_h_190,
    _mm_id_mapcount: [i32; 2],
    __slate_anon_5: __SlateRecord_mm_types_h_465,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_465 {
    _mm_id: [u32; 2],
    _mm_ids: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_483 {
    __slate_anon_0: __SlateRecord_mm_types_h_484,
    __page_2: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_484 {
    _flags_2: u64,
    _head_2: u64,
    _deferred_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_497 {
    __slate_anon_0: __SlateRecord_mm_types_h_498,
    __page_3: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_498 {
    _flags_3: u64,
    _head_3: u64,
    _hugetlb_subpool: *mut (),
    _hugetlb_cgroup: *mut (),
    _hugetlb_cgroup_rsvd: *mut (),
    _hugetlb_hwpoison: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


// get_task_state():
// set_current_state() includes a barrier so that the write of current->__state
// is correctly serialised wrt the caller's subsequent test of whether to
// actually sleep:
//
//   for (;;) {
// 	set_current_state(TASK_UNINTERRUPTIBLE);
// 	if (CONDITION)
// 	   break;
//
// 	schedule();
//   }
//   __set_current_state(TASK_RUNNING);
//
// If the caller does not need such serialisation (because, for instance, the
// CONDITION test and condition change and wakeup are under the same lock) then
// use __set_current_state().
//
// The above is typically ordered against the wakeup, which does:
//
//   CONDITION = 1;
//   wake_up_state(p, TASK_UNINTERRUPTIBLE);
//
// where wake_up_state()/try_to_wake_up() executes a full memory barrier before
// accessing p->__state.
//
// Wakeup will do: if (@state & p->__state) p->__state = TASK_RUNNING, that is,
// once it observes the TASK_UNINTERRUPTIBLE store the waking CPU can issue a
// TASK_RUNNING store which can collide with __set_current_state(TASK_RUNNING).
//
// However, with slightly different timing the wakeup TASK_RUNNING store can
// also collide with the TASK_UNINTERRUPTIBLE store. Losing that store is not
// a problem either because that will result in one extra go around the loop
// and our @cond test will save the day.
//
// Also see the comments of try_to_wake_up().
// set_special_state() should be used for those states when the blocking task
// can not use the regular condition based wait-loop. In that case we must
// serialize against wakeups such that any possible in-flight TASK_RUNNING
// stores will not collide with our state change.
// may shadow
#[repr(C)]
#[derive(Clone, Copy)]
struct prev_cputime {
    utime: u64,
    stime: u64,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_info {
    pcount: u64,
    run_delay: u64,
    max_run_delay: u64,
    min_run_delay: u64,
    last_arrival: u64,
    last_queued: u64,
    max_run_delay_ts: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_830 {
    remap: __SlateRecord_mm_types_h_831,
    simple_ioremap: __SlateRecord_mm_types_h_837,
    map_kernel: __SlateRecord_mm_types_h_841,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_831 {
    start: u64,
    start_pfn: u64,
    size: u64,
    pgprot: pgprot,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_837 {
    start_phys_addr: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_841 {
    start: u64,
    pages: *mut *mut page,
    nr_pages: u64,
    pgoff: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_avg {
    last_update_time: u64,
    load_sum: u64,
    runnable_sum: u64,
    util_sum: u32,
    period_contrib: u32,
    load_avg: u64,
    runnable_avg: u64,
    util_avg: u64,
    util_est: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_statistics {
    wait_start: u64,
    wait_max: u64,
    wait_count: u64,
    wait_sum: u64,
    iowait_count: u64,
    iowait_sum: u64,
    sleep_start: u64,
    sleep_max: u64,
    sum_sleep_runtime: i64,
    block_start: u64,
    block_max: u64,
    sum_block_runtime: i64,
    exec_max: i64,
    slice_max: u64,
    nr_failed_migrations_affine: u64,
    nr_failed_migrations_running: u64,
    nr_failed_migrations_hot: u64,
    nr_forced_migrations: u64,
    nr_wakeups: u64,
    nr_wakeups_sync: u64,
    nr_wakeups_migrate: u64,
    nr_wakeups_local: u64,
    nr_wakeups_remote: u64,
    nr_wakeups_affine: u64,
    nr_wakeups_affine_attempts: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_926 {
    __slate_anon_0: __SlateRecord_mm_types_h_927,
    vm_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_927 {
    vm_start: u64,
    vm_end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_947 {
    vm_flags: u64,
    flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_entity {
    load: load_weight,
    h_load: load_weight,
    run_node: rb_node,
    deadline: u64,
    min_vruntime: u64,
    min_slice: u64,
    max_slice: u64,
    group_node: list_head,
    on_rq: u8,
    sched_delayed: u8,
    rel_deadline: u8,
    custom_slice: u8,
    exec_start: u64,
    sum_exec_runtime: u64,
    prev_sum_exec_runtime: u64,
    vruntime: u64,
    vlag: i64,
    vprot: u64,
    slice: u64,
    nr_migrations: u64,
    depth: i32,
    parent: *mut sched_entity,
    cfs_rq: *mut cfs_rq,
    my_q: *mut cfs_rq,
    runnable_weight: u64,
    avg: sched_avg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1176 {
    __slate_anon_0: __SlateRecord_mm_types_h_1181,
    mm_mt: maple_tree,
    mmap_base: u64,
    mmap_legacy_base: u64,
    mmap_compat_base: u64,
    mmap_compat_legacy_base: u64,
    task_size: u64,
    pgd: *mut __SlateRecord_pgtable_types_h_295,
    membarrier_state: __SlateRecord_types_h_190,
    mm_users: __SlateRecord_types_h_190,
    mm_cid: mm_mm_cid,
    sched_cache_grp: *mut sched_cache_group,
    pgtables_bytes: __SlateRecord_types_h_197,
    map_count: i32,
    page_table_lock: spinlock,
    mmap_lock: rw_semaphore,
    mmlist: list_head,
    vma_writer_wait: rcuwait,
    mm_lock_seq: seqcount,
    futex: futex_mm_data,
    hiwater_rss: u64,
    hiwater_vm: u64,
    total_vm: u64,
    locked_vm: u64,
    pinned_vm: __SlateRecord_types_h_197,
    data_vm: u64,
    exec_vm: u64,
    stack_vm: u64,
    __slate_anon_28: __SlateRecord_mm_types_h_1289,
    write_protect_seq: seqcount,
    arg_lock: spinlock,
    start_code: u64,
    end_code: u64,
    start_data: u64,
    end_data: u64,
    start_brk: u64,
    brk: u64,
    start_stack: u64,
    arg_start: u64,
    arg_end: u64,
    env_start: u64,
    env_end: u64,
    saved_auxv: [u64; 56],
    rss_stat: [percpu_counter; 4],
    binfmt: *mut linux_binfmt,
    context: __SlateRecord_mmu_h_84,
    flags: __SlateRecord_mm_types_h_1171,
    ioctx_lock: spinlock,
    ioctx_table: *mut kioctx_table,
    exe_file: *mut file,
    notifier_subscriptions: *mut mmu_notifier_subscriptions,
    tlb_flush_pending: __SlateRecord_types_h_190,
    tlb_flush_batched: __SlateRecord_types_h_190,
    uprobes_state: uprobes_state,
    hugetlb_usage: __SlateRecord_types_h_197,
    async_put_work: work_struct,
    iommu_mm: *mut iommu_mm_data,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1181 {
    mm_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_1289 {
    def_flags: u64,
    def_vma_flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct linux_binfmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_notifier_subscriptions {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_rt_entity {
    run_list: list_head,
    timeout: u64,
    watchdog_stamp: u64,
    time_slice: u32,
    on_rq: u16,
    on_list: u16,
    back: *mut sched_rt_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_dl_entity {
    rb_node: rb_node,
    dl_runtime: u64,
    dl_deadline: u64,
    dl_period: u64,
    dl_bw: u64,
    dl_density: u64,
    runtime: i64,
    deadline: u64,
    flags: u32,
    __slate_bits_0: __slate_bits::__SlateBits296U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_58 {
    __ubuf_iovec: iovec,
    __slate_anon_1: __SlateRecord_uio_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uio_h_65 {
    __slate_anon_0: __SlateRecord_uio_h_66,
    count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_66 {
    __iov: *const iovec,
    kvec: *const kvec,
    bvec: *const bio_vec,
    folioq: *const folio_queue,
    xarray: *mut xarray,
    ubuf: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_struct {
    thread_info: thread_info,
    __state: u32,
    saved_state: u32,
    stack: *mut (),
    usage: refcount_struct,
    flags: u32,
    ptrace: u32,
    on_cpu: u8,
    on_rq: u8,
    is_blocked: u8,
    __pad: u8,
    wake_entry: __call_single_node,
    wakee_flips: u32,
    wakee_flip_decay_ts: u64,
    last_wakee: *mut task_struct,
    recent_used_cpu: i32,
    wake_cpu: i32,
    prio: i32,
    static_prio: i32,
    normal_prio: i32,
    rt_priority: u32,
    se: sched_entity,
    rt: sched_rt_entity,
    dl: sched_dl_entity,
    dl_server: *mut sched_dl_entity,
    sched_class: *const sched_class,
    sched_task_group: *mut task_group,
    stats: sched_statistics,
    btrace_seq: u32,
    policy: u32,
    max_allowed_capacity: u64,
    nr_cpus_allowed: i32,
    cpus_ptr: *const cpumask,
    user_cpus_ptr: *mut cpumask,
    cpus_mask: cpumask,
    migration_pending: *mut (),
    migration_disabled: u16,
    migration_flags: u16,
    rcu_read_lock_nesting: i32,
    rcu_read_unlock_special: rcu_special,
    rcu_node_entry: list_head,
    rcu_blocked_node: *mut rcu_node,
    rcu_tasks_nvcsw: u64,
    rcu_tasks_holdout: u8,
    rcu_tasks_idx: u8,
    rcu_tasks_idle_cpu: i32,
    rcu_tasks_holdout_list: list_head,
    rcu_tasks_exit_cpu: i32,
    rcu_tasks_exit_list: list_head,
    trc_reader_nesting: i32,
    trc_reader_scp: *mut srcu_ctr,
    sched_info: sched_info,
    tasks: list_head,
    pushable_tasks: plist_node,
    pushable_dl_tasks: rb_node,
    mm: *mut mm_struct,
    active_mm: *mut mm_struct,
    exec_state: *mut task_exec_state,
    exit_state: i32,
    exit_code: i32,
    exit_signal: i32,
    pdeath_signal: i32,
    jobctl: u64,
    personality: u32,
    __slate_bits_0: __slate_bits::__SlateBits65U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits65U1,
    atomic_flags: u64,
    restart_block: restart_block,
    pid: i32,
    tgid: i32,
    stack_canary: u64,
    real_parent: *mut task_struct,
    parent: *mut task_struct,
    children: list_head,
    sibling: list_head,
    group_leader: *mut task_struct,
    ptraced: list_head,
    ptrace_entry: list_head,
    thread_pid: *mut pid,
    pid_links: [hlist_node; 4],
    thread_node: list_head,
    vfork_done: *mut completion,
    set_child_tid: *mut i32,
    clear_child_tid: *mut i32,
    worker_private: *mut (),
    utime: u64,
    stime: u64,
    gtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    start_time: u64,
    start_boottime: u64,
    min_flt: u64,
    maj_flt: u64,
    posix_cputimers: posix_cputimers,
    posix_cputimers_work: posix_cputimers_work,
    ptracer_cred: *const cred,
    real_cred: *const cred,
    cred: *const cred,
    cached_requested_key: *mut key,
    comm: [u8; 16],
    nameidata: *mut nameidata,
    sysvsem: sysv_sem,
    sysvshm: sysv_shm,
    real_fs: *mut fs_struct,
    fs: *mut fs_struct,
    files: *mut files_struct,
    io_uring: *mut io_uring_task,
    io_uring_restrict: *mut io_restriction,
    nsproxy: *mut nsproxy,
    signal: *mut signal_struct,
    sighand: *mut sighand_struct,
    blocked: __SlateRecord_signal_h_25,
    real_blocked: __SlateRecord_signal_h_25,
    saved_sigmask: __SlateRecord_signal_h_25,
    pending: sigpending,
    sas_ss_sp: u64,
    sas_ss_size: u64,
    sas_ss_flags: u32,
    task_works: *mut callback_head,
    audit_context: *mut audit_context,
    loginuid: __SlateRecord_uidgid_types_h_9,
    sessionid: u32,
    seccomp: seccomp,
    syscall_dispatch: syscall_user_dispatch,
    parent_exec_id: u64,
    self_exec_id: u64,
    alloc_lock: spinlock,
    pi_lock: raw_spinlock,
    wake_q: wake_q_node,
    pi_waiters: rb_root_cached,
    pi_top_task: *mut task_struct,
    pi_blocked_on: *mut rt_mutex_waiter,
    blocked_on: *mut mutex,
    blocked_lock: raw_spinlock,
    blocked_donor: *mut task_struct,
    journal_info: *mut (),
    bio_list: *mut bio_list,
    plug: *mut blk_plug,
    reclaim_state: *mut reclaim_state,
    io_context: *mut io_context,
    capture_control: *mut capture_control,
    ptrace_message: u64,
    last_siginfo: *mut kernel_siginfo,
    ioac: task_io_accounting,
    acct_rss_mem1: u64,
    acct_vm_mem1: u64,
    acct_timexpd: u64,
    mems_allowed: __SlateRecord_nodemask_types_h_17,
    mems_allowed_seq: seqcount_spinlock,
    cpuset_mem_spread_rotor: i32,
    cgroups: *mut css_set,
    cg_list: list_head,
    futex: futex_sched_data,
    perf_recursion: [u8; 4],
    perf_event_ctxp: *mut perf_event_context,
    perf_event_mutex: mutex,
    perf_event_list: list_head,
    perf_ctx_data: *mut perf_ctx_data,
    ipi_mask: task_ipi_mask,
    mempolicy: *mut mempolicy,
    il_prev: i16,
    il_weight: u8,
    pref_node_fork: i16,
    cache_work: callback_head,
    sched_cache_grp: *mut sched_cache_group,
    preferred_llc: i32,
    pref_llc_queued: i32,
    rseq: rseq_data,
    mm_cid: sched_mm_cid,
    tlb_ubc: tlbflush_unmap_batch,
    splice_pipe: *mut pipe_inode_info,
    task_frag: page_frag,
    delays: *mut task_delay_info,
    nr_dirtied: i32,
    nr_dirtied_pause: i32,
    dirty_paused_when: u64,
    timer_slack_ns: u64,
    default_timer_slack_ns: u64,
    trace_recursion: u64,
    throttle_disk: *mut gendisk,
    utask: *mut uprobe_task,
    kmap_ctrl: kmap_ctrl,
    rcu: callback_head,
    rcu_users: refcount_struct,
    pagefault_disabled: i32,
    oom_reaper_list: *mut task_struct,
    oom_reaper_timer: timer_list,
    stack_vm_area: *mut vm_struct,
    stack_refcount: refcount_struct,
    security: *mut (),
    bpf_net_context: *mut bpf_net_context,
    mce_vaddr: *mut (),
    mce_kflags: u64,
    mce_addr: u64,
    __slate_bits_2: __slate_bits::__SlateBits65U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


// Per process flags
// I'm a virtual CPU
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// All allocations inherit GFP_NOFS. See memalloc_nfs_save()
// All allocations inherit GFP_NOIO. See memalloc_noio_save()
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Allocations constrained to zones which allow long term pinning.
// See memalloc_pin_save()
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
fn test_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_ti_thread_flag(task_thread_info!(tsk), flag)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_cache_group {
    pcpu_sched: *mut sched_cache_time,
    lock: raw_spinlock,
    epoch: u64,
    nr_running_avg: u64,
    next_scan: u64,
    footprint: u64,
    cpu: i32,
    refcnt: refcount_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Dependency order vs. p above.
// Dependency order vs. p above.
// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
// @v: The value to statically initialize with.
// rcu_assign_pointer() - assign to RCU-protected pointer
// @p: pointer to assign to
// @v: value to assign (publish)
//
// Assigns the specified value to the specified RCU-protected
// pointer, ensuring that any concurrent RCU readers will see
// any prior initialization.
//
// Inserts memory barriers on architectures that require them
// (which is most of them), and also prevents the compiler from
// reordering the code that initializes the structure after the pointer
// assignment.  More importantly, this call documents which pointers
// will be dereferenced by RCU read-side code.
//
// In some special cases, you may use RCU_INIT_POINTER() instead
// of rcu_assign_pointer().  RCU_INIT_POINTER() is a bit faster due
// to the fact that it does not constrain either the CPU or the compiler.
// That said, using RCU_INIT_POINTER() when you should have used
// rcu_assign_pointer() is a very bad thing that results in
// impossible-to-diagnose memory corruption.  So please be careful.
// See the RCU_INIT_POINTER() comment header for details.
//
// Note that rcu_assign_pointer() evaluates each of its arguments only
// once, appearances notwithstanding.  One of the "extra" evaluations
// is in typeof() and the other visible only to sparse (__CHECKER__),
// neither of which actually execute the argument.  As with most cpp
// macros, this execute-arguments-only-once property is important, so
// please be careful when making changes to rcu_assign_pointer() and the
// other macros that it invokes.
// rcu_replace_pointer() - replace an RCU pointer, returning its old value
// @rcu_ptr: RCU pointer, whose old value is returned
// @ptr: regular pointer
// @c: the lockdep conditions under which the dereference will take place
//
// Perform a replacement, where @rcu_ptr is an RCU-annotated
// pointer and @c is the lockdep argument that is passed to the
// rcu_dereference_protected() call used to read that pointer.  The old
// value of @rcu_ptr is returned, and @rcu_ptr is set to @ptr.
// rcu_access_pointer() - fetch RCU pointer with no dereferencing
// @p: The pointer to read
//
// Return the value of the specified RCU-protected pointer, but omit the
// lockdep checks for being in an RCU read-side critical section.  This is
// useful when the value of this pointer is accessed, but the pointer is
// not dereferenced, for example, when testing an RCU-protected pointer
// against NULL.  Within an RCU read-side critical section, there is little
// reason to use rcu_access_pointer().  Although rcu_access_pointer() may
// also be used in cases where update-side locks prevent the value of the
// pointer from changing, you should instead use rcu_dereference_protected()
// for this use case.  It is also permissible to use rcu_access_pointer()
// within lockless updaters to obtain the old value for an atomic operation,
// for example, for cmpxchg().
//
// It is usually best to test the rcu_access_pointer() return value
// directly in order to avoid accidental dereferences being introduced
// by later inattentive changes.  In other words, assigning the
// rcu_access_pointer() return value to a local variable results in an
// accident waiting to happen.
//
// It is also permissible to use rcu_access_pointer() when read-side
// access to the pointer was removed at least one grace period ago, as is
// the case in the context of the RCU callback that is freeing up the data,
// or after a synchronize_rcu() returns.  This can be useful when tearing
// down multi-linked structures after a grace period has elapsed.  However,
// rcu_dereference_protected() is normally preferred for this use case.
// rcu_dereference_check() - rcu_dereference with debug checking
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Do an rcu_dereference(), but check that the conditions under which the
// dereference will take place are correct.  Typically the conditions
// indicate the various locking conditions that should be held at that
// point.  The check should return true if the conditions are satisfied.
// An implicit check for being in an RCU read-side critical section
// (rcu_read_lock()) is included.
//
// For example:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock));
//
// could be used to indicate to lockdep that foo->bar may only be dereferenced
// if either rcu_read_lock() is held, or that the lock required to replace
// the bar struct at foo->bar is held.
//
// Note that the list of conditions may also include indications of when a lock
// need not be held, for example during initialisation or destruction of the
// target struct:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock) ||
// 					      atomic_read(&foo->usage) == 0);
//
// Inserts memory barriers on architectures that require them
// (currently only the Alpha), prevents the compiler from refetching
// (and from merging fetches), and, more importantly, documents exactly
// which pointers are protected by RCU and checks that the pointer is
// annotated as __rcu.
// rcu_dereference_protected() - fetch RCU pointer when updates prevented
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Return the value of the specified RCU-protected pointer, but omit
// the READ_ONCE().  This is useful in cases where update-side locks
// prevent the value of the pointer from changing.  Please note that this
// primitive does *not* prevent the compiler from repeating this reference
// or combining it with other references, so it should not be used without
// protection of appropriate locks.
//
// This function is only for update-side use.  Using this function
// when protected only by rcu_read_lock() will result in infrequent
// but very ugly failures.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
}

fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_117 {
    __slate_anon_0: __SlateRecord_key_h_118,
    x: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_118 {
    desc_len: u16,
    desc: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_198 {
    graveyard_link: list_head,
    serial_node: rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_user {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_208 {
    expiry: i64,
    revoked_at: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_246 {
    index_key: keyring_index_key,
    __slate_anon_1: __SlateRecord_key_h_248,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_248 {
    hash: u64,
    len_desc: u64,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_261 {
    payload: key_payload,
    __slate_anon_1: __SlateRecord_key_h_263,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_263 {
    name_link: list_head,
    keys: assoc_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstat {
    result_mask: u32,
    mode: u16,
    nlink: u32,
    blksize: u32,
    attributes: u64,
    attributes_mask: u64,
    ino: u64,
    dev: u32,
    rdev: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    size: i64,
    atime: timespec64,
    mtime: timespec64,
    ctime: timespec64,
    btime: timespec64,
    blocks: u64,
    mnt_id: u64,
    change_cookie: u64,
    subvol: u64,
    dio_mem_align: u32,
    dio_offset_align: u32,
    dio_read_offset_align: u32,
    atomic_write_unit_min: u32,
    atomic_write_unit_max: u32,
    atomic_write_unit_max_opt: u32,
    atomic_write_segments_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


// wait_queue_entry::flags
#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


fn init_waitqueue_entry(wq_entry: &mut wait_queue_entry, p: *mut task_struct) {
unsafe {
(*(wq_entry as *mut wait_queue_entry)).flags = 0;
}
unsafe {
(*(wq_entry as *mut wait_queue_entry)).private = p as *mut ();
}
unsafe {
(*(wq_entry as *mut wait_queue_entry)).func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>>(default_wake_function as *const ()) };
}
}

// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
}


static class_rwsem_read_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rwsem_h_259 {
    lock: *mut rw_semaphore,
}


#[inline(always)]
fn class_rwsem_read_destructor(_T: *mut __SlateRecord_rwsem_h_259) {
unsafe { up_read((*_T).lock) };
}

#[inline(always)]
fn class_rwsem_read_lock_ptr(_T: &mut __SlateRecord_rwsem_h_259) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_rwsem_h_259)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_rwsem_read_constructor(l: *mut rw_semaphore) -> __SlateRecord_rwsem_h_259 {
let mut _t: __SlateRecord_rwsem_h_259 = __SlateRecord_rwsem_h_259 { lock: l };
let _T: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(_t);
unsafe { down_read((*_T).lock) };
return _t;
}

#[inline(always)]
fn __class_rwsem_read_cleanup_ctx(_T: *mut *mut __SlateRecord_rwsem_h_259) {
}

static class_rwsem_write_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rwsem_h_270 {
    lock: *mut rw_semaphore,
}


#[inline(always)]
fn class_rwsem_write_destructor(_T: *mut __SlateRecord_rwsem_h_270) {
unsafe { up_write((*_T).lock) };
}

#[inline(always)]
fn class_rwsem_write_lock_ptr(_T: &mut __SlateRecord_rwsem_h_270) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_rwsem_h_270)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_rwsem_write_constructor(l: *mut rw_semaphore) -> __SlateRecord_rwsem_h_270 {
let mut _t: __SlateRecord_rwsem_h_270 = __SlateRecord_rwsem_h_270 { lock: l };
let _T: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(_t);
unsafe { down_write((*_T).lock) };
return _t;
}

#[inline(always)]
fn __class_rwsem_write_cleanup_ctx(_T: *mut *mut __SlateRecord_rwsem_h_270) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_segcblist {
    head: *mut callback_head,
    tails: [*mut *mut callback_head; 4],
    gp_seq: [rcu_gp_seq; 4],
    len: i64,
    seglen: [i64; 4],
    flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct srcu_data {
    srcu_ctrs: [srcu_ctr; 2],
    srcu_reader_flavor: i32,
    __slate_pad_2: [u8; 28],
    lock: raw_spinlock,
    srcu_cblist: rcu_segcblist,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_cblist_invoking: bool,
    delay_work: timer_list,
    work: work_struct,
    srcu_barrier_head: callback_head,
    srcu_ec_head: callback_head,
    srcu_ec_state: i32,
    mynode: *mut srcu_node,
    grpmask: u64,
    cpu: i32,
    ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_node {
    lock: raw_spinlock,
    srcu_have_cbs: [u64; 4],
    srcu_data_have_cbs: [u64; 4],
    srcu_gp_seq_needed_exp: u64,
    srcu_parent: *mut srcu_node,
    grplo: i32,
    grphi: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_usage {
    node: *mut srcu_node,
    level: [*mut srcu_node; 3],
    srcu_size_state: i32,
    srcu_cb_mutex: mutex,
    lock: raw_spinlock,
    srcu_gp_mutex: mutex,
    srcu_gp_seq: u64,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_gp_start: u64,
    srcu_last_gp_end: u64,
    srcu_size_jiffies: u64,
    srcu_n_lock_retries: u64,
    srcu_n_exp_nodelay: u64,
    sda_is_static: bool,
    srcu_barrier_seq: u64,
    srcu_barrier_mutex: mutex,
    srcu_barrier_completion: completion,
    srcu_barrier_cpu_cnt: __SlateRecord_types_h_190,
    reschedule_jiffies: u64,
    reschedule_count: u64,
    work: delayed_work,
    irq_work: irq_work,
    srcu_ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct export_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_sb_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstatfs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mount {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mtd_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_lockref_h_26 {
    lock_count: u64,
    __slate_anon_1: __SlateRecord_lockref_h_30,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_lockref_h_30 {
    lock: spinlock,
    count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_task {
    state: u32,
    depth: u32,
    return_instances: *mut return_instance,
    ri_pool: *mut return_instance,
    ri_timer: timer_list,
    ri_seqcount: seqcount,
    __slate_anon_6: __SlateRecord_uprobes_h_136,
    active_uprobe: *mut uprobe,
    xol_vaddr: u64,
    signal_denied: bool,
    auprobe: *mut arch_uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_50 {
    __slate_anon_0: __SlateRecord_dcache_h_51,
    hash_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dcache_h_51 {
    hash: u32,
    len: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct return_instance {
    hprobe: hprobe,
    func: u64,
    stack: u64,
    orig_ret_vaddr: u64,
    chained: bool,
    cons_cnt: i32,
    next: *mut return_instance,
    rcu: callback_head,
    consumer: return_consumer,
    extra_consumers: *mut return_consumer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_99 {
    __d_name: qstr,
    d_name: qstr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_126 {
    d_alias: hlist_node,
    d_in_lookup_hash: hlist_bl_node,
    d_rcu: callback_head,
    waiters: *mut completion_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mmu_h_84 {
    ctx_id: u64,
    tlb_gen: __SlateRecord_types_h_197,
    next_trim_cpumask: u64,
    ldt_usr_sem: rw_semaphore,
    ldt: *mut ldt_struct,
    flags: u64,
    lock: mutex,
    vdso: *mut (),
    vdso_image: *const vdso_image,
    perf_rdpmc_allowed: __SlateRecord_types_h_190,
    pkey_allocation_map: u16,
    execute_only_pkey: i16,
    global_asid: u16,
    asid_transition: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct page {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_89,
    __slate_anon_2: __SlateRecord_mm_types_h_155,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_mark_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_kattr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits436U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_272 {
    ia_uid: __SlateRecord_uidgid_types_h_9,
    ia_vfsuid: __SlateRecord_mnt_idmapping_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_276 {
    ia_gid: __SlateRecord_uidgid_types_h_13,
    ia_vfsgid: __SlateRecord_mnt_idmapping_h_21,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
}


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_area_desc {
    mm: *mut mm_struct,
    file: *mut file,
    start: u64,
    end: u64,
    pgoff: u64,
    vm_file: *mut file,
    vma_flags: __SlateRecord_mm_types_h_871,
    page_prot: pgprot,
    vm_ops: *const vm_operations_struct,
    private_data: *mut (),
    action: mmap_action,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct vm_area_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_926,
    vm_mm: *mut mm_struct,
    vm_page_prot: pgprot,
    __slate_anon_3: __SlateRecord_mm_types_h_947,
    vm_lock_seq: u32,
    __vm_anon_pgoff_lo: u32,
    anon_vma_chain: list_head,
    anon_vma: *mut anon_vma,
    vm_ops: *const vm_operations_struct,
    vm_pgoff: u64,
    vm_file: *mut file,
    vm_private_data: *mut (),
    swap_readahead_info: __SlateRecord_types_h_197,
    vm_policy: *mut mempolicy,
    __slate_pad_14: [u8; 8],
    vm_refcnt: refcount_struct,
    __vm_anon_pgoff_hi: u32,
    shared: __SlateRecord_mm_types_h_1062,
    vm_userfaultfd_ctx: vm_userfaultfd_ctx,
    pfnmap_track_ctx: *mut pfnmap_track_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_790 {
    i_nlink: u32,
    __i_nlink: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_834 {
    i_dentry: hlist_head,
    i_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_846 {
    i_fop: *const file_operations,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_852 {
    i_devices: list_head,
    i_linklen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_856 {
    i_pipe: *mut pipe_inode_info,
    i_cdev: *mut cdev,
    i_link: *mut u8,
    i_dir_seq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
}


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Clear the access bit
// Clear the dirty bit
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1267 {
    f_path: path,
    __f_path: path,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1271 {
    f_pos_lock: mutex,
    f_pipe: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1287 {
    f_task_work: callback_head,
    f_llist: llist_node,
    f_ra: file_ra_state,
    f_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lease {
}


fn gfpflags_allow_blocking(gfp_flags: u32) -> bool {
gfp_flags & (BIT!(10i32) as u32) != 0
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


fn check_object_size(ptr: *const (), n: u64, to_user: bool) {
}

fn copy_overflow(size: i32, count: u64) {
if IS_ENABLED__2!() != 0 {
unsafe { __copy_overflow(size, count) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[inline(always)]
fn check_copy_size(addr: *const (), bytes: u64, is_source: bool) -> bool {
let sz: i32 = (unsafe { __slate_intrinsic_2913c0303de876b3(addr, const { false as bool }, const { false as bool }, const { false as bool }) }) as i32;
if unlikely!(sz >= 0 && (sz as u64) < bytes) != 0 {
if !unsafe { __slate_intrinsic_f217c4720f04f6db(bytes) } {
copy_overflow(sz, bytes);
} else {
if is_source {
unsafe { __bad_copy_from() };
} else {
unsafe { __bad_copy_to() };
}
}
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (bytes > 2147483647u64) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("204: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 204b - ., 3; .popsection\n# {0}", const 204 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_check_copy_size_1, sym __slate_asm_string_check_copy_size_2, const 57 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("205: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 205b - ., 4; .popsection\n# {0}", const 205 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return false;
}
check_object_size(addr, bytes, is_source);
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iov_iter {
    iter_type: u8,
    nofault: bool,
    data_source: bool,
    iov_offset: u64,
    __slate_anon_4: __SlateRecord_uio_h_58,
    __slate_anon_5: __SlateRecord_uio_h_78,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrinker {
    count_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    scan_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    batch: i64,
    seeks: i32,
    flags: u32,
    refcount: refcount_struct,
    done: completion,
    rcu: callback_head,
    private_data: *mut (),
    list: list_head,
    nr_deferred: *mut __SlateRecord_types_h_197,
}


fn fs_reclaim_acquire(gfp_mask: u32) {
}

fn fs_reclaim_release(gfp_mask: u32) {
}

fn might_alloc(gfp_mask: u32) {
fs_reclaim_acquire(gfp_mask);
fs_reclaim_release(gfp_mask);
if (unsafe { (*get_current()).flags }) & 2048 != 0 {
return;
}
might_sleep_if!(gfpflags_allow_blocking(gfp_mask));
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_107 {
    is_visible: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, i32) -> u16>,
    is_visible_const: Option<unsafe extern "C" fn(*mut kobject, *const attribute, i32) -> u16>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_118 {
    attrs: *mut *mut attribute,
    attrs_const: *const *const attribute,
}


fn xa_insert(xa: *mut xarray, index: u64, entry: *mut (), gfp: u32) -> i32 {
let mut err: i32 = 0;
might_alloc(gfp);
xa_lock!(xa);
err = unsafe { __xa_insert(xa, index, entry, gfp) };
xa_unlock!(xa);
err
}

// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
#[inline(always)]
fn copy_from_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(to as *const (), n, false) {
return n;
}
unsafe { _copy_from_user(to, from, n) }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[inline(always)]
fn copy_to_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(from, n, true) {
return n;
}
unsafe { _copy_to_user(to, from, n) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key {
    usage: refcount_struct,
    serial: i32,
    __slate_anon_2: __SlateRecord_key_h_198,
    sem: rw_semaphore,
    user: *mut key_user,
    security: *mut (),
    __slate_anon_6: __SlateRecord_key_h_208,
    last_used_at: i64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    perm: u32,
    quotalen: u16,
    datalen: u16,
    state: i16,
    flags: u64,
    __slate_anon_15: __SlateRecord_key_h_246,
    __slate_anon_16: __SlateRecord_key_h_261,
    restrict_link: *mut key_restriction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_struct {
    __count: refcount_struct,
    epoll_watches: percpu_counter,
    unix_inflight: u64,
    pipe_bufs: __SlateRecord_types_h_197,
    uidhash_node: hlist_node,
    uid: __SlateRecord_uidgid_types_h_9,
    locked_vm: __SlateRecord_types_h_197,
    ratelimit: ratelimit_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_symbol {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_sect_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_notes_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cred {
    usage: __SlateRecord_types_h_197,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    suid: __SlateRecord_uidgid_types_h_9,
    sgid: __SlateRecord_uidgid_types_h_13,
    euid: __SlateRecord_uidgid_types_h_9,
    egid: __SlateRecord_uidgid_types_h_13,
    fsuid: __SlateRecord_uidgid_types_h_9,
    fsgid: __SlateRecord_uidgid_types_h_13,
    securebits: u32,
    cap_inheritable: __SlateRecord_capability_h_24,
    cap_permitted: __SlateRecord_capability_h_24,
    cap_effective: __SlateRecord_capability_h_24,
    cap_bset: __SlateRecord_capability_h_24,
    cap_ambient: __SlateRecord_capability_h_24,
    jit_keyring: u8,
    session_keyring: *mut key,
    process_keyring: *mut key,
    thread_keyring: *mut key,
    request_key_auth: *mut key,
    security: *mut (),
    user: *mut user_struct,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    group_info: *mut group_info,
    __slate_anon_25: __SlateRecord_cred_h_147,
}


// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_memremap_h_143 {
    range: range,
    __slate_anon_1: __SlateRecord_memremap_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145 {
    __empty_ranges: __SlateRecord_memremap_h_145_2,
    ranges: [range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_h_754 {
    vma: *mut vm_area_struct,
    gfp_mask: u32,
    pgoff: u64,
    address: u64,
    real_address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_h_768 {
    orig_pte: __SlateRecord_pgtable_64_types_h_21,
    orig_pmd: __SlateRecord_pgtable_64_types_h_22,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid {
    count: refcount_struct,
    level: u32,
    lock: spinlock,
    __slate_anon_3: __SlateRecord_pid_h_62,
    tasks: [hlist_head; 4],
    inodes: hlist_head,
    wait_pidfd: wait_queue_head,
    rcu: callback_head,
    numbers: [upid; 0],
}


fn get_pid(pid: *mut pid) -> *mut pid {
if !pid.is_null() {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*pid).count) });
}
pid
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


fn task_pid(task: *mut task_struct) -> *mut pid {
unsafe { (*task).thread_pid }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pacct_struct {
    ac_flag: i32,
    ac_exitcode: i64,
    ac_mem: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_state {
    nr_threads: __SlateRecord_types_h_190,
    dumper: core_thread,
    startup: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct signal_struct {
    sigcnt: refcount_struct,
    live: __SlateRecord_types_h_190,
    nr_threads: i32,
    quick_threads: i32,
    thread_head: list_head,
    wait_chldexit: wait_queue_head,
    curr_target: *mut task_struct,
    shared_pending: sigpending,
    multiprocess: hlist_head,
    group_exit_code: i32,
    notify_count: i32,
    group_exec_task: *mut task_struct,
    group_stop_count: i32,
    flags: u32,
    core_state: *mut core_state,
    __slate_bits_0: __slate_bits::__SlateBits299U0,
    next_posix_timer_id: __SlateRecord_types_h_190,
    posix_timers: hlist_head,
    ignored_posix_timers: hlist_head,
    real_timer: hrtimer,
    it_real_incr: i64,
    it: [cpu_itimer; 2],
    cputimer: thread_group_cputimer,
    posix_cputimers: posix_cputimers,
    pids: [*mut pid; 4],
    tty_old_pgrp: *mut pid,
    leader: i32,
    tty: *mut tty_struct,
    stats_lock: seqlock,
    utime: u64,
    stime: u64,
    cutime: u64,
    cstime: u64,
    gtime: u64,
    cgtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    cnvcsw: u64,
    cnivcsw: u64,
    min_flt: u64,
    maj_flt: u64,
    cmin_flt: u64,
    cmaj_flt: u64,
    inblock: u64,
    oublock: u64,
    cinblock: u64,
    coublock: u64,
    maxrss: u64,
    cmaxrss: u64,
    ioac: task_io_accounting,
    sum_sched_runtime: u64,
    rlim: [rlimit; 16],
    pacct: pacct_struct,
    stats: *mut taskstats,
    audit_tty: u32,
    tty_audit_buf: *mut tty_audit_buf,
    cgroup_threadgroup_rwsem: rw_semaphore,
    oom_flag_origin: bool,
    oom_score_adj: i16,
    oom_score_adj_min: i16,
    oom_mm: *mut mm_struct,
    cred_guard_mutex: mutex,
    exec_update_lock: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


// group exit in progress
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
}


fn signal_pending(p: *mut task_struct) -> i32 {
// TIF_NOTIFY_SIGNAL isn't really a signal, but it requires the same
// behavior in terms of ensuring that we break out of wait loops
// so that notify signal callbacks can be processed.
if unlikely__2!(test_tsk_thread_flag(p, 2i32)) != 0 {
return 1;
}
task_sigpending(p)
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_361() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
}

fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqblk {
    dqb_bhardlimit: i64,
    dqb_bsoftlimit: i64,
    dqb_curspace: i64,
    dqb_rsvspace: i64,
    dqb_ihardlimit: i64,
    dqb_isoftlimit: i64,
    dqb_curinodes: i64,
    dqb_btime: i64,
    dqb_itime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqinfo {
    dqi_format: *mut quota_format_type,
    dqi_fmt_id: i32,
    dqi_dirty_list: list_head,
    dqi_flags: u64,
    dqi_bgrace: u32,
    dqi_igrace: u32,
    dqi_max_spc_limit: i64,
    dqi_max_ino_limit: i64,
    dqi_priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_shutdown_f_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_fasync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot {
    dq_hash: hlist_node,
    dq_inuse: list_head,
    dq_free: list_head,
    dq_dirty: list_head,
    dq_lock: mutex,
    dq_dqb_lock: spinlock,
    dq_count: __SlateRecord_types_h_190,
    dq_sb: *mut super_block,
    dq_id: kqid,
    dq_off: i64,
    dq_flags: u64,
    dq_dqb: mem_dqblk,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_ops {
    check_quota_file: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    write_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    free_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    commit_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot_operations {
    write_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    alloc_dquot: Option<unsafe extern "C" fn(*mut super_block, i32) -> *mut dquot>,
    destroy_dquot: Option<unsafe extern "C" fn(*mut dquot)>,
    acquire_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    mark_dirty: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    get_reserved_space: Option<unsafe extern "C" fn(*mut inode) -> *mut i64>,
    get_projid: Option<unsafe extern "C" fn(*mut inode, *mut __SlateRecord_projid_h_24) -> i32>,
    get_inode_usage: Option<unsafe extern "C" fn(*mut inode, *mut i64) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_info_h_65 {
    text: snd_info_entry_text,
    ops: *const snd_info_entry_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_dqblk {
    d_fieldmask: i32,
    d_spc_hardlimit: u64,
    d_spc_softlimit: u64,
    d_ino_hardlimit: u64,
    d_ino_softlimit: u64,
    d_space: u64,
    d_ino_count: u64,
    d_ino_timer: i64,
    d_spc_timer: i64,
    d_ino_warns: i32,
    d_spc_warns: i32,
    d_rt_spc_hardlimit: u64,
    d_rt_spc_softlimit: u64,
    d_rt_space: u64,
    d_rt_spc_timer: i64,
    d_rt_spc_warns: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_type_state {
    flags: u32,
    spc_timelimit: u32,
    ino_timelimit: u32,
    rt_spc_timelimit: u32,
    spc_warnlimit: u32,
    ino_warnlimit: u32,
    rt_spc_warnlimit: u32,
    ino: u64,
    blocks: u64,
    nextents: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_info {
    i_fieldmask: i32,
    i_flags: u32,
    i_spc_timelimit: u32,
    i_ino_timelimit: u32,
    i_rt_spc_timelimit: u32,
    i_spc_warnlimit: u32,
    i_ino_warnlimit: u32,
    i_rt_spc_warnlimit: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quotactl_ops {
    quota_on: Option<unsafe extern "C" fn(*mut super_block, i32, i32, *const path) -> i32>,
    quota_off: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    quota_enable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_disable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_sync: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    set_info: Option<unsafe extern "C" fn(*mut super_block, i32, *mut qc_info) -> i32>,
    get_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_nextdqblk: Option<unsafe extern "C" fn(*mut super_block, *mut kqid, *mut qc_dqblk) -> i32>,
    set_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut super_block, *mut qc_state) -> i32>,
    rm_xquota: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1164 {
    integer: __SlateRecord_asound_h_1165,
    integer64: __SlateRecord_asound_h_1170,
    enumerated: __SlateRecord_asound_h_1175,
    reserved: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1165 {
    min: i64,
    max: i64,
    step: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1170 {
    min: i64,
    max: i64,
    step: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1175 {
    items: u32,
    item: u32,
    name: [u8; 64],
    names_ptr: u64,
    names_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1190 {
    integer: __SlateRecord_asound_h_1191,
    integer64: __SlateRecord_asound_h_1195,
    enumerated: __SlateRecord_asound_h_1199,
    bytes: __SlateRecord_asound_h_1203,
    iec958: snd_aes_iec958,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1191 {
    value: [i64; 128],
    value_ptr: *mut i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1195 {
    value: [i64; 64],
    value_ptr: *mut i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1199 {
    item: [u32; 128],
    item_ptr: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1203 {
    data: [u8; 512],
    data_ptr: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_operations {
    alloc_inode: Option<unsafe extern "C" fn(*mut super_block) -> *mut inode>,
    destroy_inode: Option<unsafe extern "C" fn(*mut inode)>,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
    dirty_inode: Option<unsafe extern "C" fn(*mut inode, i32)>,
    write_inode: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    sync_inode_metadata: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    drop_inode: Option<unsafe extern "C" fn(*mut inode) -> i32>,
    evict_inode: Option<unsafe extern "C" fn(*mut inode)>,
    put_super: Option<unsafe extern "C" fn(*mut super_block)>,
    sync_fs: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    freeze_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    freeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    thaw_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    unfreeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    statfs: Option<unsafe extern "C" fn(*mut dentry, *mut kstatfs) -> i32>,
    umount_begin: Option<unsafe extern "C" fn(*mut super_block)>,
    show_options: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_devname: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_path: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_stats: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    quota_read: Option<unsafe extern "C" fn(*mut super_block, i32, *mut u8, u64, i64) -> i64>,
    quota_write: Option<unsafe extern "C" fn(*mut super_block, i32, *const u8, u64, i64) -> i64>,
    get_dquots: Option<unsafe extern "C" fn(*mut inode) -> *mut *mut dquot>,
    nr_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    free_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    remove_bdev: Option<unsafe extern "C" fn(*mut super_block, *mut block_device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut super_block)>,
    report_error: Option<unsafe extern "C" fn(*const fserror_event)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asound_h_1265 {
    elem: __SlateRecord_asound_h_1266,
    data8: [u8; 60],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asound_h_1266 {
    mask: u32,
    id: snd_ctl_elem_id,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct super_block {
    s_list: list_head,
    s_dev: u32,
    s_super_dev: *mut super_dev,
    s_blocksize_bits: u8,
    s_blocksize: u64,
    s_maxbytes: i64,
    s_type: *mut file_system_type,
    s_op: *const super_operations,
    dq_op: *const dquot_operations,
    s_qcop: *const quotactl_ops,
    s_export_op: *const export_operations,
    s_flags: u64,
    s_iflags: u64,
    s_magic: u64,
    s_root: *mut dentry,
    s_umount: rw_semaphore,
    s_passive: refcount_struct,
    s_active: __SlateRecord_types_h_190,
    s_security: *mut (),
    s_xattr: *const *const xattr_handler,
    s_roots: hlist_head,
    s_roots_lock: spinlock,
    s_mounts: *mut mount,
    s_bdev: *mut block_device,
    s_bdev_file: *mut file,
    s_bdi: *mut backing_dev_info,
    s_mtd: *mut mtd_info,
    s_instances: hlist_node,
    s_quota_types: u32,
    s_dquot: quota_info,
    s_writers: sb_writers,
    s_fs_info: *mut (),
    s_time_gran: u32,
    s_time_min: i64,
    s_time_max: i64,
    s_fsnotify_mask: u32,
    s_fsnotify_info: *mut fsnotify_sb_info,
    s_id: [u8; 32],
    s_uuid: __SlateRecord_uuid_h_21,
    s_uuid_len: u8,
    s_sysfs_name: [u8; 37],
    s_max_links: u32,
    s_d_flags: u32,
    s_vfs_rename_mutex: mutex,
    s_subtype: *const u8,
    __s_d_op: *const dentry_operations,
    s_shrink: *mut shrinker,
    s_remove_count: __SlateRecord_types_h_197,
    s_readonly_remount: i32,
    s_wb_err: u32,
    s_dio_done_wq: *mut workqueue_struct,
    s_pins: hlist_head,
    s_user_ns: *mut user_namespace,
    s_dentry_lru: list_lru,
    s_inode_lru: list_lru,
    rcu: callback_head,
    destroy_work: work_struct,
    s_sync_lock: mutex,
    s_stack_depth: i32,
    __slate_pad_59: [u8; 12],
    s_inode_list_lock: spinlock,
    s_inodes: list_head,
    s_inode_wblist_lock: spinlock,
    s_inodes_wb: list_head,
    s_min_writeback_pages: i64,
    s_pending_errors: refcount_struct,
}


// sb->s_flags.  Note that these mirror the equivalent MS_* flags where
// represented in both.
// Mount read-only
// Update the on-disk [acm]times lazily
#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_control_h_59 {
    c: Option<unsafe extern "C" fn(*mut snd_kcontrol, i32, u32, *mut u32) -> i32>,
    p: *const u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
}


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_control_h_78 {
    c: Option<unsafe extern "C" fn(*mut snd_kcontrol, i32, u32, *mut u32) -> i32>,
    p: *const u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
    d_flags: u32,
    d_seq: seqcount_spinlock,
    d_hash: hlist_bl_node,
    d_parent: *mut dentry,
    __slate_anon_4: __SlateRecord_dcache_h_99,
    d_inode: *mut inode,
    d_shortname: shortname_store,
    d_op: *const dentry_operations,
    d_sb: *mut super_block,
    d_time: u64,
    d_fsdata: *mut (),
    d_lockref: lockref,
    d_lru: list_head,
    d_sib: hlist_node,
    d_children: hlist_head,
    __slate_anon_15: __SlateRecord_dcache_h_126,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dentry_operations {
    d_revalidate: Option<unsafe extern "C" fn(*mut inode, *const qstr, *mut dentry, u32) -> i32>,
    d_weak_revalidate: Option<unsafe extern "C" fn(*mut dentry, u32) -> i32>,
    d_hash: Option<unsafe extern "C" fn(*const dentry, *mut qstr) -> i32>,
    d_compare: Option<unsafe extern "C" fn(*const dentry, u32, *const u8, *const qstr) -> i32>,
    d_delete: Option<unsafe extern "C" fn(*const dentry) -> i32>,
    d_init: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
    d_release: Option<unsafe extern "C" fn(*mut dentry)>,
    d_prune: Option<unsafe extern "C" fn(*mut dentry)>,
    d_iput: Option<unsafe extern "C" fn(*mut dentry, *mut inode)>,
    d_dname: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> *mut u8>,
    d_automount: Option<unsafe extern "C" fn(*mut path) -> *mut vfsmount>,
    d_manage: Option<unsafe extern "C" fn(*const path, bool) -> i32>,
    d_real: Option<unsafe extern "C" fn(*mut dentry, u32) -> *mut dentry>,
    d_unalias_trylock: Option<unsafe extern "C" fn(*const dentry) -> bool>,
    d_unalias_unlock: Option<unsafe extern "C" fn(*const dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_31 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_31_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_31_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_31_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_31_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_31_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_239_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_299 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_299_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_784 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_784_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_889 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_889_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_890 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_890_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_891 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_891_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_892 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_892_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_893 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_893_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_894 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_894_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_1566 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_1566_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_1566_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_1931 {
    op: i32,
    perm: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_1940 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_1944 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_control_compat_c_56 {
    integer: __SlateRecord_control_compat_c_57,
    integer64: __SlateRecord_control_compat_c_62,
    enumerated: __SlateRecord_control_compat_c_67,
    reserved: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_compat_c_57 {
    min: i32,
    max: i32,
    step: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_compat_c_62 {
    min: u64,
    max: u64,
    step: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_compat_c_67 {
    items: u32,
    item: u32,
    name: [u8; 64],
    names_ptr: u64,
    names_length: u32,
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(mnt as *const vfsmount)).mnt_idmap) as *const *mut mnt_idmap) };
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_control_compat_c_143 {
    integer: [i32; 128],
    data: [u8; 512],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_2580 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_control_c_2580_2 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits296U0 {
        #[bits(1)]
        pub dl_throttled: u32,
        #[bits(1)]
        pub dl_yielded: u32,
        #[bits(1)]
        pub dl_non_contending: u32,
        #[bits(1)]
        pub dl_overrun: u32,
        #[bits(1)]
        pub dl_server: u32,
        #[bits(1)]
        pub dl_server_active: u32,
        #[bits(1)]
        pub dl_defer: u32,
        #[bits(1)]
        pub dl_defer_armed: u32,
        #[bits(1)]
        pub dl_defer_running: u32,
        #[bits(1)]
        pub dl_defer_idle: u32,
        #[bits(1)]
        pub dl_bw_attached: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits65U0 {
        #[bits(1)]
        pub sched_reset_on_fork: u32,
        #[bits(1)]
        pub sched_contributes_to_load: u32,
        #[bits(1)]
        pub sched_migrated: u32,
        #[bits(1)]
        pub sched_task_hot: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits65U1 {
        #[bits(1)]
        pub sched_remote_wakeup: u32,
        #[bits(1)]
        pub sched_rt_mutex: u32,
        #[bits(1)]
        pub in_execve: u32,
        #[bits(1)]
        pub in_iowait: u32,
        #[bits(1)]
        pub restore_sigmask: u32,
        #[bits(1)]
        pub no_cgroup_migration: u32,
        #[bits(1)]
        pub frozen: u32,
        #[bits(1)]
        pub use_memdelay: u32,
        #[bits(1)]
        pub in_eventfd: u32,
        #[bits(1)]
        pub pasid_activated: u32,
        #[bits(1)]
        pub reported_split_lock: u32,
        #[bits(1)]
        pub in_thrashing: u32,
        #[bits(1)]
        pub in_nf_duplicate: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits299U0 {
        #[bits(1)]
        pub is_child_subreaper: u32,
        #[bits(1)]
        pub has_child_subreaper: u32,
        #[bits(1)]
        pub autoreap: u32,
        #[bits(1)]
        pub timer_create_restore_ids: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits65U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits144U0 {
        #[bits(8)]
        pub base1: u16,
        #[bits(4)]
        pub r#type: u16,
        #[bits(1)]
        pub s: u16,
        #[bits(2)]
        pub dpl: u16,
        #[bits(1)]
        pub p: u16,
        #[bits(4)]
        pub limit1: u16,
        #[bits(1)]
        pub avl: u16,
        #[bits(1)]
        pub l: u16,
        #[bits(1)]
        pub d: u16,
        #[bits(1)]
        pub g: u16,
        #[bits(8)]
        pub base2: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits146U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits588U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits656U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits599U0 {
        #[bits(1)]
        pub state_initialized: u32,
        #[bits(1)]
        pub state_in_sysfs: u32,
        #[bits(1)]
        pub state_add_uevent_sent: u32,
        #[bits(1)]
        pub state_remove_uevent_sent: u32,
        #[bits(1)]
        pub uevent_suppress: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits145U0 {
        #[bits(4)]
        pub sp_reg: u32,
        #[bits(4)]
        pub bp_reg: u32,
        #[bits(3)]
        pub r#type: u32,
        #[bits(1)]
        pub signal: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits436U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits218U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U0 {
        #[bits(1)]
        pub can_wakeup: bool,
        #[bits(1)]
        pub async_suspend: bool,
        #[bits(1)]
        pub in_dpm_list: bool,
        #[bits(1)]
        pub is_prepared: bool,
        #[bits(1)]
        pub is_suspended: bool,
        #[bits(1)]
        pub is_noirq_suspended: bool,
        #[bits(1)]
        pub is_late_suspended: bool,
        #[bits(1)]
        pub no_pm: bool,
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub direct_complete: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits706U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U1 {
        #[bits(1)]
        pub wakeup_path: bool,
        #[bits(1)]
        pub syscore: bool,
        #[bits(1)]
        pub no_pm_callbacks: bool,
        #[bits(1)]
        pub smart_suspend: bool,
        #[bits(1)]
        pub must_resume: bool,
        #[bits(1)]
        pub may_skip_resume: bool,
        #[bits(1)]
        pub out_band_wakeup: bool,
        #[bits(1)]
        pub strict_midlayer: bool,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U2 {
        #[bits(3)]
        pub disable_depth: u32,
        #[bits(1)]
        pub idle_notification: bool,
        #[bits(1)]
        pub request_pending: bool,
        #[bits(1)]
        pub deferred_resume: bool,
        #[bits(1)]
        pub needs_force_resume: bool,
        #[bits(1)]
        pub runtime_auto: bool,
        #[bits(1)]
        pub ignore_children: bool,
        #[bits(1)]
        pub no_callbacks: bool,
        #[bits(1)]
        pub irq_safe: bool,
        #[bits(1)]
        pub use_autosuspend: bool,
        #[bits(1)]
        pub timer_autosuspends: bool,
        #[bits(1)]
        pub memalloc_noio: bool,
        #[bits(2, access = na)]
        pub __slate_pad_12: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits709U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits771U0 {
        #[bits(1)]
        pub indirect: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits588U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


// no-op
#[inline(always)]
fn __free_kfree(p: *mut ()) {
let _T: *mut () = unsafe { *(p as *mut *mut ()) };
if !IS_ERR_OR_NULL(_T as *const ()) {
unsafe { kfree(_T as *const ()) };
}
}

// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
#[inline(always)]
fn kmalloc_type(flags: u32, token: __SlateRecord_slab_h_532) -> u32 {
// The most common case is KMALLOC_NORMAL, so test for it
// with a single branch for all the relevant flags.
if likely!(flags & ((BIT!(4i32) as u32) | if IS_ENABLED__2!() != 0 { BIT!(0i32) as u32 } else { 0 } | if IS_ENABLED!() != 0 { BIT!(22i32) as u32 } else { 0 }) == 0) != 0 {
return 0;
}
// At least one of the flags has to be set. Their priorities in
// decreasing order are:
//  1) __GFP_DMA
//  2) __GFP_RECLAIMABLE
//  3) __GFP_ACCOUNT
if IS_ENABLED__2!() != 0 && flags & (BIT!(0i32) as u32) != 0 {
return 2;
}
if IS_ENABLED!() == 0 || flags & (BIT!(4i32) as u32) != 0 {
return 1;
} else {
return 0;
}
}

#[inline(always)]
fn __kmalloc_index(size: u64, size_is_constant: bool) -> u32 {
if size == 0 {
return 0;
}
if size <= 8u64 {
return 3;
}
if 8i32 <= 32 && size > 64 && size <= 96 {
return 1;
}
if 8i32 <= 64 && size > 128 && size <= 192 {
return 2;
}
if size <= 8 {
return 3;
}
if size <= 16 {
return 4;
}
if size <= 32 {
return 5;
}
if size <= 64 {
return 6;
}
if size <= 128 {
return 7;
}
if size <= 256 {
return 8;
}
if size <= 512 {
return 9;
}
if size <= 1024 {
return 10;
}
if size <= 2048u64 {
return 11;
}
if size <= 4096u64 {
return 12;
}
if size <= 8192u64 {
return 13;
}
if size <= 16384u64 {
return 14;
}
if size <= 32768u64 {
return 15;
}
if size <= 65536u64 {
return 16;
}
if size <= 131072u64 {
return 17;
}
if size <= 262144u64 {
return 18;
}
if size <= 524288u64 {
return 19;
}
if size <= 1048576u64 {
return 20;
}
if size <= 2097152u64 {
return 21;
}
if IS_ENABLED!() == 0 && size_is_constant {
loop {
if true {
unsafe { __compiletime_assert_392() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 3; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___kmalloc_index_1, sym __slate_asm_string___kmalloc_index_2, const 834 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
// Will never be reached. Needed because the compiler may complain
1i32.wrapping_neg() as u32
}

#[inline(always)]
fn _kmalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_noprof(size, flags) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, size) };
}
unsafe { __kmalloc_noprof(PASS_TOKEN_PARAMS!(size), flags) }
}

// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// __alloc_flex - Allocate an object that has a trailing flexible array
// @KMALLOC: kmalloc wrapper function to use for allocation.
// @GFP: GFP flags for the allocation.
// @TYPE: type of structure to allocate space for.
// @FAM: The name of the flexible array member of @TYPE structure.
// @COUNT: how many @FAM elements to allocate space for.
//
// Returns: Newly allocated pointer to @TYPE with @COUNT-many trailing
// @FAM elements, or NULL on failure or if @COUNT cannot be represented
// by the member of @TYPE that counts the @FAM elements (annotated via
// __counted_by()).
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[inline(always)]
fn __free_kvfree(p: *mut ()) {
let _T: *mut () = unsafe { *(p as *mut *mut ()) };
if !IS_ERR_OR_NULL(_T as *const ()) {
unsafe { kvfree(_T as *const ()) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


// File is embedded in backing_file object
// File is capable of returning -EAGAIN if I/O will block
// The two FMODE_NONOTIFY* define which fsnotify events should not be generated
// for an open file. These are the possible values of
// (f->f_mode & FMODE_FSNOTIFY_MASK) and their meaning:
//
// FMODE_NONOTIFY - suppress all (incl. non-permission) events.
// FMODE_NONOTIFY_PERM - suppress permission (incl. pre-content) events.
// FMODE_NONOTIFY | FMODE_NONOTIFY_PERM - suppress only FAN_ACCESS_PERM.
// Whiteout is represented by a char device.  The following constants define the
// mode and device number to use.
#[repr(C)]
#[derive(Clone, Copy)]
struct iattr {
    ia_valid: u32,
    ia_mode: u16,
    __slate_anon_2: __SlateRecord_fs_h_272,
    __slate_anon_3: __SlateRecord_fs_h_276,
    ia_size: i64,
    ia_atime: timespec64,
    ia_mtime: timespec64,
    ia_ctime: timespec64,
    ia_file: *mut file,
}


// non-RWF related bits - start at 16
#[repr(C)]
#[derive(Clone, Copy)]
struct kiocb {
    ki_filp: *mut file,
    ki_pos: i64,
    ki_complete: Option<unsafe extern "C" fn(*mut kiocb, i64)>,
    private: *mut (),
    ki_flags: i32,
    ki_ioprio: u16,
    ki_write_stream: u8,
    ki_waitq: *mut wait_page_queue,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space_operations {
    read_folio: Option<unsafe extern "C" fn(*mut file, *mut folio) -> i32>,
    writepages: Option<unsafe extern "C" fn(*mut address_space, *mut writeback_control) -> i32>,
    dirty_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> bool>,
    readahead: Option<unsafe extern "C" fn(*mut readahead_control)>,
    write_begin: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, *mut *mut folio, *mut *mut ()) -> i32>,
    write_end: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, u32, *mut folio, *mut ()) -> i32>,
    bmap: Option<unsafe extern "C" fn(*mut address_space, u64) -> u64>,
    invalidate_folio: Option<unsafe extern "C" fn(*mut folio, u64, u64)>,
    release_folio: Option<unsafe extern "C" fn(*mut folio, u32) -> bool>,
    free_folio: Option<unsafe extern "C" fn(*mut folio)>,
    direct_IO: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    migrate_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio, *mut folio, u32) -> i32>,
    launder_folio: Option<unsafe extern "C" fn(*mut folio) -> i32>,
    is_partially_uptodate: Option<unsafe extern "C" fn(*mut folio, u64, u64) -> bool>,
    is_dirty_writeback: Option<unsafe extern "C" fn(*mut folio, *mut bool, *mut bool)>,
    error_remove_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> i32>,
    swap_activate: Option<unsafe extern "C" fn(*mut swap_info_struct, *mut file, *mut u64) -> i32>,
    swap_deactivate: Option<unsafe extern "C" fn(*mut file)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
    host: *mut inode,
    i_pages: xarray,
    invalidate_lock: rw_semaphore,
    gfp_mask: u32,
    i_mmap_writable: __SlateRecord_types_h_190,
    i_mmap: rb_root_cached,
    nrpages: u64,
    writeback_index: u64,
    a_ops: *const address_space_operations,
    flags: u64,
    wb_err: u32,
    i_private_lock: spinlock,
    i_mmap_rwsem: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode {
    i_mode: u16,
    i_opflags: u16,
    i_flags: u32,
    i_acl: *mut posix_acl,
    i_default_acl: *mut posix_acl,
    i_uid: __SlateRecord_uidgid_types_h_9,
    i_gid: __SlateRecord_uidgid_types_h_13,
    i_op: *const inode_operations,
    i_sb: *mut super_block,
    i_mapping: *mut address_space,
    i_security: *mut (),
    i_ino: u64,
    __slate_anon_12: __SlateRecord_fs_h_790,
    i_rdev: u32,
    i_size: i64,
    i_atime_sec: i64,
    i_mtime_sec: i64,
    i_ctime_sec: i64,
    i_atime_nsec: u32,
    i_mtime_nsec: u32,
    i_ctime_nsec: u32,
    i_generation: u32,
    i_lock: spinlock,
    i_bytes: u16,
    i_blkbits: u8,
    i_write_hint: u8,
    i_blocks: u64,
    i_state: inode_state_flags,
    i_rwsem: rw_semaphore,
    dirtied_when: u64,
    dirtied_time_when: u64,
    i_hash: hlist_node,
    i_io_list: list_head,
    i_lru: list_head,
    i_sb_list: list_head,
    i_wb_list: list_head,
    __slate_anon_36: __SlateRecord_fs_h_834,
    i_version: __SlateRecord_types_h_197,
    i_sequence: __SlateRecord_types_h_197,
    i_count: __SlateRecord_types_h_190,
    i_dio_count: __SlateRecord_types_h_190,
    i_writecount: __SlateRecord_types_h_190,
    i_readcount: __SlateRecord_types_h_190,
    __slate_anon_43: __SlateRecord_fs_h_846,
    i_flctx: *mut file_lock_context,
    i_data: address_space,
    __slate_anon_46: __SlateRecord_fs_h_852,
    __slate_anon_47: __SlateRecord_fs_h_856,
    i_fsnotify_mask: u32,
    i_fsnotify_marks: *mut fsnotify_mark_connector,
    i_private: *mut (),
}


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
fn iminor(inode: *const inode) -> u32 {
MINOR!(unsafe { (*inode).i_rdev })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fown_struct {
    file: *mut file,
    lock: rwlock,
    pid: *mut pid,
    pid_type: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    euid: __SlateRecord_uidgid_types_h_9,
    signum: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_ra_state {
    start: u64,
    size: u32,
    async_size: u32,
    ra_pages: u32,
    order: u16,
    mmap_miss: u16,
    prev_pos: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file {
    f_lock: spinlock,
    f_mode: u32,
    f_op: *const file_operations,
    f_mapping: *mut address_space,
    private_data: *mut (),
    f_inode: *mut inode,
    f_flags: u32,
    f_iocb_flags: u32,
    f_cred: *const cred,
    f_owner: *mut fown_struct,
    __slate_anon_10: __SlateRecord_fs_h_1267,
    __slate_anon_11: __SlateRecord_fs_h_1271,
    f_pos: i64,
    f_security: *mut (),
    f_wb_err: u32,
    f_sb_err: u32,
    f_ep: *mut hlist_head,
    __slate_anon_17: __SlateRecord_fs_h_1287,
    f_ref: __SlateRecord_file_ref_h_52,
}


// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
#[repr(C)]
#[derive(Clone, Copy)]
struct dir_context {
    actor: Option<unsafe extern "C" fn(*mut dir_context, *const u8, i32, i64, u64, u32) -> bool>,
    pos: i64,
    count: i32,
    dt_flags_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_operations {
    owner: *mut module,
    fop_flags: u32,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    write_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    iopoll: Option<unsafe extern "C" fn(*mut kiocb, *mut io_comp_batch, u32) -> i32>,
    iterate_shared: Option<unsafe extern "C" fn(*mut file, *mut dir_context) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    flush: Option<unsafe extern "C" fn(*mut file, *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    fsync: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    fasync: Option<unsafe extern "C" fn(i32, *mut file, i32) -> i32>,
    lock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    check_flags: Option<unsafe extern "C" fn(i32) -> i32>,
    flock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    splice_write: Option<unsafe extern "C" fn(*mut pipe_inode_info, *mut file, *mut i64, u64, u32) -> i64>,
    splice_read: Option<unsafe extern "C" fn(*mut file, *mut i64, *mut pipe_inode_info, u64, u32) -> i64>,
    splice_eof: Option<unsafe extern "C" fn(*mut file)>,
    setlease: Option<unsafe extern "C" fn(*mut file, i32, *mut *mut file_lease, *mut *mut ()) -> i32>,
    fallocate: Option<unsafe extern "C" fn(*mut file, i32, i64, i64) -> i64>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut seq_file, *mut file)>,
    copy_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, u64, u32) -> i64>,
    remap_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, i64, u32) -> i64>,
    fadvise: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    uring_cmd: Option<unsafe extern "C" fn(*mut io_uring_cmd, u32) -> i32>,
    uring_cmd_iopoll: Option<unsafe extern "C" fn(*mut io_uring_cmd, *mut io_comp_batch, u32) -> i32>,
    mmap_prepare: Option<unsafe extern "C" fn(*mut vm_area_desc) -> i32>,
}


// File system supports uncached read/write buffered IO
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct inode_operations {
    lookup: Option<unsafe extern "C" fn(*mut inode, *mut dentry, u32) -> *mut dentry>,
    get_link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut delayed_call) -> *const u8>,
    permission: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, i32) -> i32>,
    get_inode_acl: Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>,
    readlink: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> i32>,
    create: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> i32>,
    link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut dentry) -> i32>,
    unlink: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    symlink: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *const u8) -> i32>,
    mkdir: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> *mut dentry>,
    rmdir: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    mknod: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16, u32) -> i32>,
    rename: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *mut inode, *mut dentry, u32) -> i32>,
    setattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>,
    getattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>,
    listxattr: Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>,
    fiemap: Option<unsafe extern "C" fn(*mut inode, *mut fiemap_extent_info, u64, u64) -> i32>,
    update_time: Option<unsafe extern "C" fn(*mut inode, u32, u32) -> i32>,
    sync_lazytime: Option<unsafe extern "C" fn(*mut inode)>,
    atomic_open: Option<unsafe extern "C" fn(*mut inode, *mut dentry, *mut file, u32, u16) -> i32>,
    tmpfile: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut file, u16) -> i32>,
    get_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, i32) -> *mut posix_acl>,
    set_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>,
    fileattr_set: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut file_kattr) -> i32>,
    fileattr_get: Option<unsafe extern "C" fn(*mut dentry, *mut file_kattr) -> i32>,
    get_offset_ctx: Option<unsafe extern "C" fn(*mut inode) -> *mut offset_ctx>,
}


// no suid or xattr security attributes
// Note that nosuid etc flags are inode-specific: setting some file-system
// flags just means all the inodes inherit those flags by default. It might be
// possible to override it selectively if you really wanted to with some
// ioctl() that is not currently implemented.
//
// Exception: SB_RDONLY is always applied to the entire file system.
//
// Unfortunately, it is possible to change a filesystems flags with it mounted
// with files in use.  This means that all of the inodes will not have their
// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
#[repr(C)]
#[derive(Clone, Copy)]
struct file_system_type {
    name: *const u8,
    fs_flags: i32,
    init_fs_context: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    parameters: *const fs_parameter_spec,
    kill_sb: Option<unsafe extern "C" fn(*mut super_block)>,
    owner: *mut module,
    list: hlist_node,
    fs_supers: hlist_head,
    s_lock_key: lock_class_key,
    s_umount_key: lock_class_key,
    s_vfs_rename_key: lock_class_key,
    s_writers_key: [lock_class_key; 3],
    i_lock_key: lock_class_key,
    i_mutex_key: lock_class_key,
    invalidate_lock_key: lock_class_key,
    i_mutex_dir_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
}


fn compat_ptr(uptr: u32) -> *mut () {
(uptr as u64) as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
    data: *mut (),
    size: u64,
    alt: u64,
    alt_len: u64,
    extable_base: u64,
    extable_len: u64,
    extable: *const (),
    sym___kernel_sigreturn: i64,
    sym___kernel_rt_sigreturn: i64,
    sym___kernel_vsyscall: i64,
    sym_int80_landing_pad: i64,
    sym_vdso32_sigreturn_landing_pad: i64,
    sym_vdso32_rt_sigreturn_landing_pad: i64,
    sym___futex_list64_try_unlock_cs_start: i64,
    sym___futex_list64_try_unlock_cs_end: i64,
    sym___futex_list32_try_unlock_cs_start: i64,
    sym___futex_list32_try_unlock_cs_end: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elf64_sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_node {
    count: __SlateRecord_types_h_190,
    active: __SlateRecord_types_h_190,
    __parent: *mut kernfs_node,
    name: *const u8,
    rb: rb_node,
    ns: *const ns_common,
    hash: u32,
    flags: u16,
    mode: u16,
    __slate_anon_9: __SlateRecord_kernfs_h_221,
    id: u64,
    r#priv: *mut (),
    iattr: *mut kernfs_iattrs,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_file {
    kn: *mut kernfs_node,
    file: *mut file,
    seq_file: *mut seq_file,
    r#priv: *mut (),
    mutex: mutex,
    prealloc_mutex: mutex,
    event: i32,
    list: list_head,
    prealloc_buf: *mut u8,
    atomic_write_len: u64,
    __slate_bits_0: __slate_bits::__SlateBits656U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_ops {
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    read: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    atomic_write_len: u64,
    prealloc: bool,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    mmap: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut vm_area_struct) -> i32>,
    llseek: Option<unsafe extern "C" fn(*mut kernfs_open_file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_ns_type_operations {
    r#type: u32,
    current_may_mount: Option<unsafe extern "C" fn() -> bool>,
    grab_current_ns: Option<unsafe extern "C" fn() -> *mut ns_common>,
    netlink_ns: Option<unsafe extern "C" fn(*mut sock) -> *const ns_common>,
    initial_ns: Option<unsafe extern "C" fn() -> *const ns_common>,
    drop_ns: Option<unsafe extern "C" fn(*mut ns_common)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute_group {
    name: *const u8,
    __slate_anon_1: __SlateRecord_sysfs_h_107,
    is_bin_visible: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u16>,
    bin_size: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u64>,
    __slate_anon_4: __SlateRecord_sysfs_h_118,
    bin_attrs: *const *const bin_attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bin_attribute {
    attr: attribute,
    size: u64,
    private: *mut (),
    f_mapping: Option<unsafe extern "C" fn() -> *mut address_space>,
    read: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, i64, i32) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
// buffer for the variables
#[repr(C)]
#[derive(Clone, Copy)]
struct kobject {
    name: *const u8,
    entry: list_head,
    parent: *mut kobject,
    kset: *mut kset,
    ktype: *const kobj_type,
    sd: *mut kernfs_node,
    kref: kref,
    __slate_bits_0: __slate_bits::__SlateBits599U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_type {
    release: Option<unsafe extern "C" fn(*mut kobject)>,
    sysfs_ops: *const sysfs_ops,
    default_groups: *mut *const attribute_group,
    child_ns_type: Option<unsafe extern "C" fn(*const kobject) -> *const kobj_ns_type_operations>,
    namespace: Option<unsafe extern "C" fn(*const kobject) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const kobject, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_uevent_env {
    argv: [*mut u8; 3],
    envp: [*mut u8; 64],
    envp_idx: i32,
    buf: [u8; 2048],
    buflen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset_uevent_ops {
    filter: Option<unsafe extern "C" fn(*const kobject) -> i32>,
    name: Option<unsafe extern "C" fn(*const kobject) -> *const u8>,
    uevent: Option<unsafe extern "C" fn(*const kobject, *mut kobj_uevent_env) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_notify: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_notify_one: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_new1: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_free_one: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_add: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_replace: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_remove: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_remove_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_activate_id: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_rename_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_rename: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_find_numid: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_find_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_register_ioctl: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_register_ioctl_compat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_unregister_ioctl: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_unregister_ioctl_compat: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_get_preferred_subdevice: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_request_layer: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_register_layer: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_disconnect_layer: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_boolean_mono_info: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_boolean_stereo_info: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_ctl_enum_info: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous", sym snd_ctl_notify, sym snd_ctl_notify_one, sym snd_ctl_new1, sym snd_ctl_free_one, sym snd_ctl_add, sym snd_ctl_replace, sym snd_ctl_remove, sym snd_ctl_remove_id, sym snd_ctl_activate_id, sym snd_ctl_rename_id, sym snd_ctl_rename, sym snd_ctl_find_numid, sym snd_ctl_find_id, sym snd_ctl_register_ioctl, sym snd_ctl_register_ioctl_compat, sym snd_ctl_unregister_ioctl, sym snd_ctl_unregister_ioctl_compat, sym snd_ctl_get_preferred_subdevice, sym snd_ctl_request_layer, sym snd_ctl_register_layer, sym snd_ctl_disconnect_layer, sym snd_ctl_boolean_mono_info, sym snd_ctl_boolean_stereo_info, sym snd_ctl_enum_info, options(att_syntax));

// One for each parameter, describing how to use it.  Some files do
// multiple of these per line, so can't just use MODULE_INFO.
#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param_ops {
    flags: u32,
    set: Option<unsafe extern "C" fn(*const u8, *const kernel_param) -> i32>,
    get: Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>,
    free: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param {
    name: *const u8,
    r#mod: *mut module,
    ops: *const kernel_param_ops,
    perm: u16,
    level: i8,
    flags: u8,
    __slate_anon_6: __SlateRecord_moduleparam_h_89,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_array {
    max: u32,
    elemsize: u32,
    num: *mut u32,
    ops: *const kernel_param_ops,
    elem: *mut (),
}


// module_param_named - typesafe helper for a renamed module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// Usually it's a good idea to have variable names and user-exposed names the
// same, but that's harder if the variable must be non-static or is inside a
// structure.  This allows exposure under a different name.
// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// All the helper functions
//
// The macros to do compile-time type checking stolen from Jakub
// Jelinek, who IIRC came up with this idea for the 2.4 module init code.
#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct its_array {
    pages: *mut *mut (),
    num: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_arch_specific {
    num_orcs: u32,
    orc_unwind_ip: *mut i32,
    orc_unwind: *mut orc_entry,
    its_pages: its_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_kobject {
    kobj: kobject,
    r#mod: *mut module,
    drivers_dir: *mut kobject,
    mp: *mut module_param_attrs,
    kobj_completion: *mut completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *const u8, u64) -> i64>,
    setup: Option<unsafe extern "C" fn(*mut module, *const u8)>,
    test: Option<unsafe extern "C" fn(*mut module) -> i32>,
    free: Option<unsafe extern "C" fn(*mut module)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
}


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct module {
    state: u32,
    list: list_head,
    name: [u8; 56],
    mkobj: module_kobject,
    modinfo_attrs: *mut module_attribute,
    version: *const u8,
    srcversion: *const u8,
    imported_namespaces: *const u8,
    holders_dir: *mut kobject,
    syms: *const kernel_symbol,
    crcs: *const u32,
    flagstab: *const u8,
    num_syms: u32,
    param_lock: mutex,
    kp: *mut kernel_param,
    num_kp: u32,
    using_gplonly_symbols: bool,
    async_probe_requested: bool,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    init: Option<unsafe extern "C" fn() -> i32>,
    __slate_pad_21: [u8; 8],
    mem: [module_memory; 7],
    arch: mod_arch_specific,
    taints: u64,
    num_bugs: u32,
    bug_list: list_head,
    bug_table: *mut bug_entry,
    kallsyms: *mut mod_kallsyms,
    core_kallsyms: mod_kallsyms,
    sect_attrs: *mut module_sect_attrs,
    notes_attrs: *mut module_notes_attrs,
    percpu: *mut (),
    percpu_size: u32,
    noinstr_text_start: *mut (),
    noinstr_text_size: u32,
    num_tracepoints: u32,
    tracepoints_ptrs: *const i32,
    num_srcu_structs: u32,
    srcu_struct_ptrs: *mut *mut srcu_struct,
    jump_entries: *mut jump_entry,
    num_jump_entries: u32,
    num_trace_bprintk_fmt: u32,
    trace_bprintk_fmt_start: *mut *const u8,
    trace_events: *mut *mut trace_event_call,
    num_trace_events: u32,
    trace_evals: *mut *mut trace_eval_map,
    num_trace_evals: u32,
    kprobes_text_start: *mut (),
    kprobes_text_size: u32,
    kprobe_blacklist: *mut u64,
    num_kprobe_blacklist: u32,
    num_static_call_sites: i32,
    static_call_sites: *mut static_call_site,
    source_list: list_head,
    target_list: list_head,
    exit: Option<unsafe extern "C" fn()>,
    refcnt: __SlateRecord_types_h_190,
}


// reset direct map and flush TLB on unmap, can't be freed in atomic context
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
    __slate_anon_0: __SlateRecord_vmalloc_h_53,
    addr: *mut (),
    size: u64,
    flags: u64,
    pages: *mut *mut page,
    page_order: u32,
    nr_pages: u64,
    phys_addr: u64,
    caller: *const (),
    requested_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vmem_altmap {
    base_pfn: u64,
    end_pfn: u64,
    reserve: u64,
    free: u64,
    align: u64,
    alloc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
    altmap: vmem_altmap,
    r#ref: percpu_ref,
    done: completion,
    r#type: u32,
    flags: u32,
    vmemmap_shift: u64,
    ops: *const dev_pagemap_ops,
    owner: *mut (),
    nr_range: i32,
    __slate_anon_9: __SlateRecord_memremap_h_143,
}


// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_fault {
    __slate_anon_0: __SlateRecord_mm_h_754,
    flags: u32,
    pmd: *mut __SlateRecord_pgtable_64_types_h_22,
    pud: *mut __SlateRecord_pgtable_types_h_368,
    __slate_anon_4: __SlateRecord_mm_h_768,
    cow_page: *mut page,
    page: *mut page,
    pte: *mut __SlateRecord_pgtable_64_types_h_21,
    ptl: *mut spinlock,
    prealloc_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
    open: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    close: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    mapped: Option<unsafe extern "C" fn(u64, u64, u64, *const file, *mut *mut ()) -> i32>,
    may_split: Option<unsafe extern "C" fn(*mut vm_area_struct, u64) -> i32>,
    mremap: Option<unsafe extern "C" fn(*mut vm_area_struct) -> i32>,
    mprotect: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, u64, u64) -> i32>,
    fault: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    huge_fault: Option<unsafe extern "C" fn(*mut vm_fault, u32) -> u32>,
    map_pages: Option<unsafe extern "C" fn(*mut vm_fault, u64, u64) -> u32>,
    pagesize: Option<unsafe extern "C" fn(*mut vm_area_struct) -> u64>,
    page_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    pfn_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    access: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut (), i32, i32) -> i32>,
    name: Option<unsafe extern "C" fn(*mut vm_area_struct) -> *const u8>,
    set_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, *mut mempolicy) -> i32>,
    get_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut u64) -> *mut mempolicy>,
}


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// 127: arbitrary random number, small enough to assemble well
// GUP_PIN_COUNTING_BIAS, and the associated functions that use it, overload
// the page's refcount so that two separate items are tracked: the original page
// reference count, and also a new count of how many pin_user_pages() calls were
// made against the page. ("gup-pinned" is another term for the latter).
//
// With this scheme, pin_user_pages() becomes special: such pages are marked as
// distinct from normal pages. As such, the unpin_user_page() call (and its
// variants) must be used in order to release gup-pinned pages.
//
// Choice of value:
//
// By making GUP_PIN_COUNTING_BIAS a power of two, debugging of page reference
// counts with respect to pin_user_pages() and unpin_user_page() becomes
// simpler, due to the fact that adding an even power of two to the page
// refcount has the effect of using only the upper N bits, for the code that
// counts up using the bias value. This means that the lower bits are left for
// the exclusive use of the original code that increments and decrements by one
// (or at least, by much smaller values than the bias value).
//
// Of course, once the lower bits overflow into the upper bits (and this is
// OK, because subtraction recovers the original values), then visual inspection
// no longer suffices to directly view the separate counts. However, for normal
// applications that don't have huge page reference counts, this won't be an
// issue.
//
// Locking: the lockless algorithm described in folio_try_get_rcu()
// provides safe operation for get_user_pages(), folio_mkclean() and
// other calls that race to set up page table entries.
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_ops {
    prepare: Option<unsafe extern "C" fn(*mut device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_idle: Option<unsafe extern "C" fn(*mut device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits709U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits709U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits709U2,
    links_count: u32,
    request: u32,
    runtime_status: i32,
    last_status: i32,
    runtime_error: i32,
    autosuspend_delay: i32,
    last_busy: u64,
    active_time: u64,
    suspended_time: u64,
    accounting_timestamp: u64,
    subsys_data: *mut pm_subsys_data,
    set_latency_tolerance: Option<unsafe extern "C" fn(*mut device, i32)>,
    qos: *mut dev_pm_qos,
    __slate_bits_3: __slate_bits::__SlateBits709U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_domain {
    ops: dev_pm_ops,
    start: Option<unsafe extern "C" fn(*mut device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut device, bool)>,
    activate: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync: Option<unsafe extern "C" fn(*mut device)>,
    dismiss: Option<unsafe extern "C" fn(*mut device)>,
    set_performance_state: Option<unsafe extern "C" fn(*mut device, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_type {
    name: *const u8,
    dev_name: *const u8,
    bus_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    drv_groups: *const *const attribute_group,
    r#match: Option<unsafe extern "C" fn(*mut device, *const device_driver) -> i32>,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device)>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    irq_get_affinity: Option<unsafe extern "C" fn(*mut device, u32) -> *const cpumask>,
    online: Option<unsafe extern "C" fn(*mut device) -> i32>,
    offline: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    num_vf: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_configure: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_cleanup: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
    driver_override: bool,
    need_parent_lock: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct class {
    name: *const u8,
    class_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    dev_uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16) -> *mut u8>,
    class_release: Option<unsafe extern "C" fn(*const class)>,
    dev_release: Option<unsafe extern "C" fn(*mut device)>,
    shutdown_pre: Option<unsafe extern "C" fn(*mut device) -> i32>,
    ns_type: *const kobj_ns_type_operations,
    namespace: Option<unsafe extern "C" fn(*const device) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const device, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_driver {
    name: *const u8,
    bus: *const bus_type,
    owner: *mut module,
    mod_name: *const u8,
    suppress_bind_attrs: bool,
    probe_type: u32,
    of_match_table: *const of_device_id,
    acpi_match_table: *const acpi_device_id,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    pm: *const dev_pm_ops,
    coredump: Option<unsafe extern "C" fn(*mut device)>,
    p: *mut driver_private,
    p_cb: __SlateRecord_driver_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_type {
    name: *const u8,
    groups: *mut *const attribute_group,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13) -> *mut u8>,
    release: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_links_info {
    suppliers: list_head,
    consumers: list_head,
    defer_sync: list_head,
    status: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_physical_location {
    panel: u32,
    vertical_position: u32,
    horizontal_position: u32,
    dock: bool,
    lid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device {
    kobj: kobject,
    parent: *mut device,
    p: *mut device_private,
    init_name: *const u8,
    r#type: *const device_type,
    bus: *const bus_type,
    driver: *mut device_driver,
    platform_data: *mut (),
    driver_data: *mut (),
    driver_override: __SlateRecord_device_h_719,
    mutex: mutex,
    links: dev_links_info,
    power: dev_pm_info,
    pm_domain: *mut dev_pm_domain,
    msi: dev_msi_info,
    dma_mask: *mut u64,
    coherent_dma_mask: u64,
    bus_dma_limit: u64,
    dma_range_map: *const bus_dma_region,
    dma_parms: *mut device_dma_parameters,
    dma_pools: list_head,
    dma_io_tlb_mem: *mut io_tlb_mem,
    archdata: dev_archdata,
    of_node: *mut device_node,
    fwnode: *mut fwnode_handle,
    numa_node: i32,
    devt: u32,
    id: u32,
    devres_lock: spinlock,
    devres_head: list_head,
    class: *const class,
    groups: *mut *const attribute_group,
    release: Option<unsafe extern "C" fn(*mut device)>,
    iommu_group: *mut iommu_group,
    iommu: *mut dev_iommu,
    physical_location: *mut device_physical_location,
    removable: u32,
    flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wakeup_source {
    name: *const u8,
    id: i32,
    entry: list_head,
    lock: spinlock,
    wakeirq: *mut wake_irq,
    timer: timer_list,
    timer_expires: u64,
    total_time: i64,
    max_time: i64,
    last_time: i64,
    start_prevent_time: i64,
    prevent_sleep_time: i64,
    event_count: u64,
    active_count: u64,
    relax_count: u64,
    expire_count: u64,
    wakeup_count: u64,
    dev: *mut device,
    __slate_bits_0: __slate_bits::__SlateBits706U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_device_ops {
    dev_free: Option<unsafe extern "C" fn(*mut snd_device) -> i32>,
    dev_register: Option<unsafe extern "C" fn(*mut snd_device) -> i32>,
    dev_disconnect: Option<unsafe extern "C" fn(*mut snd_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_device {
    list: list_head,
    card: *mut snd_card,
    state: u32,
    r#type: u32,
    device_data: *mut (),
    ops: *const snd_device_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_refcount {
    count: __SlateRecord_types_h_190,
    waiter: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_card {
    number: i32,
    id: [u8; 16],
    driver: [u8; 16],
    shortname: [u8; 32],
    longname: [u8; 80],
    irq_descr: [u8; 32],
    mixername: [u8; 80],
    components: *mut u8,
    components_alloc_size: u32,
    module: *mut module,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_card)>,
    devices: list_head,
    ctl_dev: *mut device,
    last_numid: u32,
    controls_rwsem: rw_semaphore,
    controls_rwlock: rwlock,
    controls_count: i32,
    user_ctl_alloc_size: u64,
    controls: list_head,
    ctl_files: list_head,
    ctl_numids: xarray,
    ctl_hash: xarray,
    ctl_hash_collision: bool,
    proc_root: *mut snd_info_entry,
    proc_root_link: *mut proc_dir_entry,
    files_list: list_head,
    s_f_ops: *mut snd_shutdown_f_ops,
    files_lock: spinlock,
    shutdown: i32,
    release_completion: *mut completion,
    dev: *mut device,
    card_dev: device,
    dev_groups: [*const attribute_group; 4],
    registered: bool,
    managed: bool,
    releasing: bool,
    sync_irq: i32,
    remove_sleep: wait_queue_head,
    total_pcm_alloc_bytes: u64,
    memory_mutex: mutex,
    power_state: u32,
    power_sleep: wait_queue_head,
    power_ref: snd_refcount,
    private_data_area: [u8; 0],
}


fn snd_power_unref(card: *mut snd_card) {
unsafe { snd_refcount_put(core::ptr::addr_of_mut!((*card).power_ref)) };
}

fn snd_card_unref(card: *mut snd_card) {
unsafe { put_device(core::ptr::addr_of_mut!((*card).card_dev)) };
}

#[inline(always)]
fn __free_snd_card_unref(p: *mut ()) {
let _T: *mut snd_card = unsafe { *(p as *mut *mut snd_card) };
if !_T.is_null() {
snd_card_unref(_T);
}
}

// The rest seem to be more-or-less nonstandard. Check them!
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
}


fn poll_wait(filp: *mut file, wait_address: *mut wait_queue_head, p: *mut poll_table_struct) {
if !p.is_null() && (unsafe { (*p)._qproc }) != None {
unsafe { unsafe { (*p)._qproc }.unwrap()(filp, wait_address, p) };
// This memory barrier is paired in the wq_has_sleeper().
// See the comment above prepare_to_wait(), we need to
// ensure that subsequent tests in this thread can't be
// reordered with __add_wait_queue() in _qproc() paths.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
    buf: *mut u8,
    size: u64,
    from: u64,
    count: u64,
    pad_until: u64,
    index: i64,
    read_pos: i64,
    lock: mutex,
    op: *const seq_operations,
    poll_event: i32,
    file: *const file,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_operations {
    start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_buffer {
    buffer: *mut u8,
    curr: u32,
    size: u32,
    len: u32,
    stop: i32,
    error: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry_text {
    read: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut snd_info_buffer)>,
    write: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut snd_info_buffer)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry_ops {
    open: Option<unsafe extern "C" fn(*mut snd_info_entry, u16, *mut *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut snd_info_entry, u16, *mut ()) -> i32>,
    read: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *mut u8, u64, i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *const u8, u64, i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, i64, i32) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *mut poll_table_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, u32, u64) -> i32>,
    mmap: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut inode, *mut file, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry {
    name: *const u8,
    mode: u16,
    size: i64,
    content: u16,
    c: __SlateRecord_info_h_65,
    parent: *mut snd_info_entry,
    module: *mut module,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_info_entry)>,
    p: *mut proc_dir_entry,
    access: mutex,
    children: list_head,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_aes_iec958 {
    status: [u8; 24],
    subcode: [u8; 147],
    pad: u8,
    dig_subframe: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_card_info {
    card: i32,
    pad: i32,
    id: [u8; 16],
    driver: [u8; 16],
    name: [u8; 32],
    longname: [u8; 80],
    reserved_: [u8; 16],
    mixername: [u8; 80],
    components: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_card_bytes {
    r#type: u32,
    data_allocated: u32,
    data_len: u32,
    reserved: u32,
    data: u64,
}


// boolean type
// integer type
// enumerated type
// byte array
// IEC958 (S/PDIF) setup
// 64-bit integer type
// virtual mixer device
// control value may be changed without a notification
// (1 << 3) is unused.
// TLV read is possible
// TLV write is possible
// TLV command is possible
// control does actually nothing, but may be updated
// write lock
// write lock owner
// kernel use a TLV callback
// user space element
// for further details see the ACPI and PCI power management specification
// full On
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_id {
    numid: u32,
    iface: i32,
    device: u32,
    subdevice: u32,
    name: [u8; 44],
    index: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_list {
    offset: u32,
    space: u32,
    used: u32,
    count: u32,
    pids: *mut snd_ctl_elem_id,
    reserved: [u8; 50],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_info {
    id: snd_ctl_elem_id,
    r#type: i32,
    access: u32,
    count: u32,
    owner: i32,
    value: __SlateRecord_asound_h_1164,
    reserved: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_value {
    id: snd_ctl_elem_id,
    __slate_bits_0: __slate_bits::__SlateBits771U0,
    value: __SlateRecord_asound_h_1190,
    reserved: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_tlv {
    numid: u32,
    length: u32,
    tlv: [u32; 0],
}


// element value was changed
// element info was changed
// element was added
// element TLV tree was changed
// element was removed
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_event {
    r#type: i32,
    data: __SlateRecord_asound_h_1265,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kcontrol_new {
    iface: i32,
    device: u32,
    subdevice: u32,
    name: *const u8,
    index: u32,
    access: u32,
    count: u32,
    info: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>,
    get: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    put: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    tlv: __SlateRecord_control_h_59,
    private_value: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kcontrol_volatile {
    owner: *mut snd_ctl_file,
    access: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kcontrol {
    list: list_head,
    id: snd_ctl_elem_id,
    count: u32,
    info: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>,
    get: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    put: Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_value) -> i32>,
    tlv: __SlateRecord_control_h_78,
    private_value: u64,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_kcontrol)>,
    vd: [snd_kcontrol_volatile; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kctl_event {
    list: list_head,
    id: snd_ctl_elem_id,
    mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_file {
    list: list_head,
    card: *mut snd_card,
    pid: *mut pid,
    preferred_subdevice: [i32; 2],
    change_sleep: wait_queue_head,
    read_lock: spinlock,
    fasync: *mut snd_fasync,
    subscribed: i32,
    events: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_layer_ops {
    next: *mut snd_ctl_layer_ops,
    module_name: *const u8,
    lregister: Option<unsafe extern "C" fn(*mut snd_card)>,
    ldisconnect: Option<unsafe extern "C" fn(*mut snd_card)>,
    lnotify: Option<unsafe extern "C" fn(*mut snd_card, u32, *mut snd_kcontrol, u32)>,
}


fn snd_ctl_get_ioffnum(kctl: *mut snd_kcontrol, id: *mut snd_ctl_elem_id) -> u32 {
let ioff: u32 = unsafe { (*id).numid }.wrapping_sub(unsafe { (*kctl).id.numid });
let mut __v0: u32;
let _i: u32 = ioff;
let _s: u32 = unsafe { (*kctl).count };
let mut _mask: u64 = 0;
let mut __v1: u64;
let mut __idx: u32 = _i;
let mut __sz: u32 = _s;
let mut __mask: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("cmp {1:e},{2:e}; sbb {0:r},{0:r}", lateout(reg) __t0, in(reg) __sz as u32, in(reg) __idx as u32, options(nostack, att_syntax));
__mask = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v1 = __mask;
_mask = __v1;
loop {
if ((const { core::mem::size_of::<u32>() }) as u64) > ((const { core::mem::size_of::<i64>() }) as u64) {
unsafe { __compiletime_assert_503() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<u32>() }) as u64) > ((const { core::mem::size_of::<i64>() }) as u64) {
unsafe { __compiletime_assert_504() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = ((_i as u64) & _mask) as u32;
__v0
}

fn snd_ctl_get_ioffidx(kctl: *mut snd_kcontrol, id: *mut snd_ctl_elem_id) -> u32 {
let ioff: u32 = unsafe { (*id).index }.wrapping_sub(unsafe { (*kctl).id.index });
let mut __v0: u32;
let _i: u32 = ioff;
let _s: u32 = unsafe { (*kctl).count };
let mut _mask: u64 = 0;
let mut __v1: u64;
let mut __idx: u32 = _i;
let mut __sz: u32 = _s;
let mut __mask: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("cmp {1:e},{2:e}; sbb {0:r},{0:r}", lateout(reg) __t0, in(reg) __sz as u32, in(reg) __idx as u32, options(nostack, att_syntax));
__mask = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v1 = __mask;
_mask = __v1;
loop {
if ((const { core::mem::size_of::<u32>() }) as u64) > ((const { core::mem::size_of::<i64>() }) as u64) {
unsafe { __compiletime_assert_505() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((const { core::mem::size_of::<u32>() }) as u64) > ((const { core::mem::size_of::<i64>() }) as u64) {
unsafe { __compiletime_assert_506() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = ((_i as u64) & _mask) as u32;
__v0
}

fn snd_ctl_get_ioff(kctl: *mut snd_kcontrol, id: *mut snd_ctl_elem_id) -> u32 {
if (unsafe { (*id).numid }) != 0 {
return snd_ctl_get_ioffnum(kctl, id);
} else {
return snd_ctl_get_ioffidx(kctl, id);
}
}

fn snd_ctl_build_ioff(dst_id: *mut snd_ctl_elem_id, src_kctl: *mut snd_kcontrol, offset: u32) -> *mut snd_ctl_elem_id {
unsafe {
*dst_id = (*src_kctl).id;
}
let __v1: u32 = unsafe { (*dst_id).index };
unsafe {
(*dst_id).index = __v1.wrapping_add(offset);
}
let __v4: u32 = unsafe { (*dst_id).numid };
unsafe {
(*dst_id).numid = __v4.wrapping_add(offset);
}
dst_id
}

// optional flags for follower
/// Max allocation size for user controls.
static mut max_user_ctl_alloc_size: i32 = 8388608;

fn __check_max_user_ctl_alloc_size() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(max_user_ctl_alloc_size) }
}

static __param_str_max_user_ctl_alloc_size: __SlateAlign16<[u8; 28]> = __SlateAlign16([115u8, 110u8, 100u8, 46u8, 109u8, 97u8, 120u8, 95u8, 117u8, 115u8, 101u8, 114u8, 95u8, 99u8, 116u8, 108u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 95u8, 115u8, 105u8, 122u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_max_user_ctl_alloc_size: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_max_user_ctl_alloc_size.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_control_c_31>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_control_c_31_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_control_c_31_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_control_c_31_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_control_c_31_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_control_c_31_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(max_user_ctl_alloc_size) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_507: [u8; 41] = [115, 110, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 109, 97, 120, 95, 117, 115, 101, 114, 95, 99, 116, 108, 95, 97, 108, 108, 111, 99, 95, 115, 105, 122, 101, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_508: [u8; 71] = [115, 110, 100, 46, 112, 97, 114, 109, 61, 109, 97, 120, 95, 117, 115, 101, 114, 95, 99, 116, 108, 95, 97, 108, 108, 111, 99, 95, 115, 105, 122, 101, 58, 77, 97, 120, 32, 97, 108, 108, 111, 99, 97, 116, 105, 111, 110, 32, 115, 105, 122, 101, 32, 102, 111, 114, 32, 117, 115, 101, 114, 32, 99, 111, 110, 116, 114, 111, 108, 115, 0];

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_kctl_ioctl {
    /// list of all ioctls
    list: list_head,
    fioctl: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>,
}


#[unsafe(no_mangle)]
static mut snd_ioctl_rwsem: rw_semaphore = rw_semaphore { count: __SlateRecord_types_h_197 { counter: 0i64 }, owner: __SlateRecord_types_h_197 { counter: 0i64 }, osq: optimistic_spin_queue { tail: __SlateRecord_types_h_190 { counter: 0i32 } }, wait_lock: raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } }, first_waiter: core::ptr::null_mut::<rwsem_waiter>() };

static mut snd_ctl_layer_rwsem: rw_semaphore = rw_semaphore { count: __SlateRecord_types_h_197 { counter: 0i64 }, owner: __SlateRecord_types_h_197 { counter: 0i64 }, osq: optimistic_spin_queue { tail: __SlateRecord_types_h_190 { counter: 0i32 } }, wait_lock: raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } }, first_waiter: core::ptr::null_mut::<rwsem_waiter>() };

static mut snd_control_ioctls: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(snd_control_ioctls) }, prev: unsafe { core::ptr::addr_of_mut!(snd_control_ioctls) } };

static mut snd_control_compat_ioctls: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(snd_control_compat_ioctls) }, prev: unsafe { core::ptr::addr_of_mut!(snd_control_compat_ioctls) } };

static mut snd_ctl_layer: *mut snd_ctl_layer_ops = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_open")]
extern "C" fn snd_ctl_open(inode: *mut inode, file: *mut file) -> i32 {
let mut __slate_compound_14: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_510: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_689> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_689 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_510) as *mut *mut __SlateRecord_spinlock_h_689;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_689> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_689 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_689;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ctl: core::mem::MaybeUninit<*mut snd_ctl_file> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut snd_ctl_file = core::ptr::addr_of_mut!(ctl) as *mut *mut snd_ctl_file;
let mut card: core::mem::MaybeUninit<*mut snd_card> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut snd_card = core::ptr::addr_of_mut!(card) as *mut *mut snd_card;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_write_lock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_write_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
stream_open(inode, file);
*__slate_slot_13 = snd_lookup_minor_data(iminor(inode as *const inode), 0i32) as *mut snd_card;
if (*__slate_slot_13).is_null() {
*__slate_slot_10 = 19i32.wrapping_neg();
} else {
*__slate_slot_10 = snd_card_file_add(*__slate_slot_13, file);
if *__slate_slot_10 >= 0 {
if !try_module_get((*(*__slate_slot_13)).module) {
*__slate_slot_10 = 19i32.wrapping_neg();
} else {
__obj_size = size_mul(core::mem::size_of::<snd_ctl_file>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_17 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_17)
});
}
*__slate_slot_12 = *__slate_slot_5 as *mut snd_ctl_file;
if (*__slate_slot_12).is_null() {
*__slate_slot_10 = 12i32.wrapping_neg();
module_put((*(*__slate_slot_13)).module);
} else {
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_12)).events));
loop {
__init_waitqueue_head(core::ptr::addr_of_mut!((*(*__slate_slot_12)).change_sleep), c"&ctl->change_sleep".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key));
if true {
break;
}
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_12)).read_lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_12)).read_lock) = *{
__slate_compound_14 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_14)
};
if true {
break;
}
}
(*(*__slate_slot_12)).card = *__slate_slot_13;
*__slate_slot_11 = 0;
while *__slate_slot_11 < 2 {
*((*(*__slate_slot_12)).preferred_subdevice.as_mut_ptr() as *mut i32).wrapping_offset(*__slate_slot_11 as isize) = 1i32.wrapping_neg();
let __v3: i32 = *__slate_slot_11;
*__slate_slot_11 = __v3.wrapping_add(1);
}
(*(*__slate_slot_12)).pid = get_pid(task_pid(get_current()));
(*file).private_data = *__slate_slot_12 as *mut ();
core::ptr::write(__slate_slot_1, class_write_lock_irqsave_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_13)).controls_rwlock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*(*__slate_slot_13)).controls_rwlock) as u64) as *mut __SlateRecord_spinlock_h_689);
__slate_cleanup_armed_0.set(true);
if !class_write_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_write_lock_irqsave_is_conditional {
if true {
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_12)).list), core::ptr::addr_of_mut!((*(*__slate_slot_13)).ctl_files));
}
}
__slate_cleanup_armed_0.set(false);
__class_write_lock_irqsave_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_write_lock_irqsave_destructor(__slate_slot_1 as *mut _);
snd_card_unref(*__slate_slot_13);
return 0;
}
}
snd_card_file_remove(*__slate_slot_13, file);
}
}
if !(*__slate_slot_13).is_null() {
snd_card_unref(*__slate_slot_13);
}
return *__slate_slot_10;
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

fn snd_ctl_empty_read_queue(ctl: *mut snd_ctl_file) {
let mut cread: *mut snd_kctl_event = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_511: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*ctl).read_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_511) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_512: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*ctl).read_lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_512) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
while list_empty((unsafe { core::ptr::addr_of_mut!((*ctl).events) }) as *const list_head) == 0 {
cread = snd_kctl_event!(unsafe { (*ctl).events.next });
list_del(unsafe { core::ptr::addr_of_mut!((*cread).list) });
unsafe { kfree(cread as *const ()) };
}
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_release")]
extern "C" fn snd_ctl_release(inode: *mut inode, file: *mut file) -> i32 {
let mut __UNIQUE_ID_unlock_516: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_516) as *mut *mut __SlateRecord_rwsem_h_270;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_270;
let mut __UNIQUE_ID_unlock_514: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_689> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut __SlateRecord_spinlock_h_689 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_514) as *mut *mut __SlateRecord_spinlock_h_689;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_689> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut __SlateRecord_spinlock_h_689 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_689;
let mut idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(idx) as *mut u32;
let mut control: core::mem::MaybeUninit<*mut snd_kcontrol> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut snd_kcontrol = core::ptr::addr_of_mut!(control) as *mut *mut snd_kcontrol;
let mut ctl: core::mem::MaybeUninit<*mut snd_ctl_file> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut snd_ctl_file = core::ptr::addr_of_mut!(ctl) as *mut *mut snd_ctl_file;
let mut card: core::mem::MaybeUninit<*mut snd_card> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut snd_card = core::ptr::addr_of_mut!(card) as *mut *mut snd_card;
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { class_write_lock_irqsave_destructor(__slate_cleanup_ptr_5 as *mut _) };
}
});
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { __class_write_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_8 = (*file).private_data as *mut snd_ctl_file;
(*file).private_data = core::ptr::null_mut::<()>();
*__slate_slot_9 = (*(*__slate_slot_8)).card;
core::ptr::write(__slate_slot_5, class_write_lock_irqsave_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_9)).controls_rwlock)));
__slate_cleanup_armed_5.set(true);
core::ptr::write(__slate_slot_4, (core::ptr::addr_of_mut!((*(*__slate_slot_9)).controls_rwlock) as u64) as *mut __SlateRecord_spinlock_h_689);
__slate_cleanup_armed_4.set(true);
if !class_write_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_5))).is_null() || !class_write_lock_irqsave_is_conditional {
if true {
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_8)).list));
}
}
__slate_cleanup_armed_4.set(false);
__class_write_lock_irqsave_cleanup_ctx(__slate_slot_4 as *mut _);
__slate_cleanup_armed_5.set(false);
class_write_lock_irqsave_destructor(__slate_slot_5 as *mut _);
core::ptr::write(__slate_slot_3, class_rwsem_write_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_9)).controls_rwsem)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!((*(*__slate_slot_9)).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_2.set(true);
if !class_rwsem_write_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_rwsem_write_is_conditional {
if true {
*__slate_slot_7 = list_first_entry__7!(core::ptr::addr_of_mut!((*(*__slate_slot_9)).controls));
while list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*(*__slate_slot_9)).controls), list) == 0 {
*__slate_slot_6 = 0;
while *__slate_slot_6 < (*(*__slate_slot_7)).count {
if (*(core::ptr::addr_of_mut!((*(*__slate_slot_7)).vd) as *mut snd_kcontrol_volatile).wrapping_offset(*__slate_slot_6 as isize)).owner == *__slate_slot_8 {
(*(core::ptr::addr_of_mut!((*(*__slate_slot_7)).vd) as *mut snd_kcontrol_volatile).wrapping_offset(*__slate_slot_6 as isize)).owner = core::ptr::null_mut::<snd_ctl_file>();
}
let __v1: u32 = *__slate_slot_6;
*__slate_slot_6 = __v1.wrapping_add(1);
}
*__slate_slot_7 = list_next_entry__8!(*__slate_slot_7, list);
}
}
}
__slate_cleanup_armed_2.set(false);
__class_rwsem_write_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_rwsem_write_destructor(__slate_slot_3 as *mut _);
snd_fasync_free((*(*__slate_slot_8)).fasync);
snd_ctl_empty_read_queue(*__slate_slot_8);
put_pid((*(*__slate_slot_8)).pid);
kfree(*__slate_slot_8 as *const ());
module_put((*(*__slate_slot_9)).module);
snd_card_file_remove(*__slate_slot_9, file);
return 0;
}
}

/// snd_ctl_notify - Send notification to user-space for a control change
/// @card: the card to send notification
/// @mask: the event mask, SNDRV_CTL_EVENT_*
/// @id: the ctl element id to send notification
///
/// This function adds an event record with the given id and mask, appends
/// to the list and wakes up the user-space for notification.  This can be
/// called in the atomic context.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_notify")]
extern "C" fn snd_ctl_notify(card: *mut snd_card, mask: u32, id: *mut snd_ctl_elem_id) {
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_520: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_520) as *mut *mut __SlateRecord_spinlock_h_610;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_610;
let mut __UNIQUE_ID_unlock_518: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_670> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut __SlateRecord_spinlock_h_670 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_518) as *mut *mut __SlateRecord_spinlock_h_670;
let mut __UNIQUE_ID_guard_517: core::mem::MaybeUninit<__SlateRecord_spinlock_h_670> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut __SlateRecord_spinlock_h_670 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_517) as *mut __SlateRecord_spinlock_h_670;
let mut ev: core::mem::MaybeUninit<*mut snd_kctl_event> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut snd_kctl_event = core::ptr::addr_of_mut!(ev) as *mut *mut snd_kctl_event;
let mut ctl: core::mem::MaybeUninit<*mut snd_ctl_file> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut snd_ctl_file = core::ptr::addr_of_mut!(ctl) as *mut *mut snd_ctl_file;
let __slate_cleanup_ptr_12 = __slate_slot_12;
let __slate_cleanup_armed_12 = core::cell::Cell::new(false);
let __slate_cleanup_12 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_12.get() {
unsafe { class_read_lock_irqsave_destructor(__slate_cleanup_ptr_12 as *mut _) };
}
});
let __slate_cleanup_ptr_11 = __slate_slot_11;
let __slate_cleanup_armed_11 = core::cell::Cell::new(false);
let __slate_cleanup_11 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_11.get() {
unsafe { __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_11 as *mut _) };
}
});
let __slate_cleanup_ptr_10 = __slate_slot_10;
let __slate_cleanup_armed_10 = core::cell::Cell::new(false);
let __slate_cleanup_10 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_10.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_10 as *mut _) };
}
});
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_9 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if snd_BUG_ON!(card.is_null() || id.is_null()) != 0 {
return;
} else {
if (*card).shutdown != 0 {
return;
} else {
core::ptr::write(__slate_slot_12, class_read_lock_irqsave_constructor(core::ptr::addr_of_mut!((*card).controls_rwlock)));
__slate_cleanup_armed_12.set(true);
core::ptr::write(__slate_slot_11, (core::ptr::addr_of_mut!((*card).controls_rwlock) as u64) as *mut __SlateRecord_spinlock_h_670);
__slate_cleanup_armed_11.set(true);
*__slate_slot_14 = list_first_entry__11!(core::ptr::addr_of_mut!((*card).ctl_files));
while list_entry_is_head!(*__slate_slot_14, core::ptr::addr_of_mut!((*card).ctl_files), list) == 0 {
if (*(*__slate_slot_14)).subscribed != 0 {
core::ptr::write(__slate_slot_10, class_spinlock_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_14)).read_lock)));
__slate_cleanup_armed_10.set(true);
core::ptr::write(__slate_slot_9, (core::ptr::addr_of_mut!((*(*__slate_slot_14)).read_lock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_9.set(true);
if !class_spinlock_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_10))).is_null() || !class_spinlock_is_conditional {
if true {
*__slate_slot_13 = list_first_entry__26!(core::ptr::addr_of_mut!((*(*__slate_slot_14)).events));
'__join_4: {
'__join_14: {
while list_entry_is_head!(*__slate_slot_13, core::ptr::addr_of_mut!((*(*__slate_slot_14)).events), list) == 0 {
if (*(*__slate_slot_13)).id.numid == (*id).numid {
break '__join_14;
} else {
*__slate_slot_13 = list_next_entry__19!(*__slate_slot_13, list);
}
}
__obj_size = size_mul(core::mem::size_of::<snd_kctl_event>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 2080u32, *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 2080u32, *{
__slate_compound_18 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_18)
});
}
*__slate_slot_13 = *__slate_slot_1 as *mut snd_kctl_event;
if !(*__slate_slot_13).is_null() {
(*(*__slate_slot_13)).id = *id;
(*(*__slate_slot_13)).mask = mask;
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_13)).list), core::ptr::addr_of_mut!((*(*__slate_slot_14)).events));
break '__join_4;
} else {
while false {
}
_dev_err((*card).dev as *const device, c"No memory available to allocate event\n".as_ptr() as *const u8);
break '__join_4;
}
}
let __v8: *mut snd_kctl_event = *__slate_slot_13;
(*__v8).mask = (*__v8).mask | mask;
}
wake_up!(core::ptr::addr_of_mut!((*(*__slate_slot_14)).change_sleep));
}
}
__slate_cleanup_armed_9.set(false);
__class_spinlock_cleanup_ctx(__slate_slot_9 as *mut _);
__slate_cleanup_armed_10.set(false);
class_spinlock_destructor(__slate_slot_10 as *mut _);
snd_kill_fasync((*(*__slate_slot_14)).fasync, 29i32, 1i32);
}
*__slate_slot_14 = list_next_entry__14!(*__slate_slot_14, list);
}
__slate_cleanup_armed_11.set(false);
__class_read_lock_irqsave_cleanup_ctx(__slate_slot_11 as *mut _);
__slate_cleanup_armed_12.set(false);
class_read_lock_irqsave_destructor(__slate_slot_12 as *mut _);
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_notify_521: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, u32, *mut snd_ctl_elem_id)>, *mut ()>(Some(snd_ctl_notify)) };

/// snd_ctl_notify_one - Send notification to user-space for a control change
/// @card: the card to send notification
/// @mask: the event mask, SNDRV_CTL_EVENT_*
/// @kctl: the pointer with the control instance
/// @ioff: the additional offset to the control index
///
/// This function calls snd_ctl_notify() and does additional jobs
/// like LED state changes.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_notify_one")]
extern "C" fn snd_ctl_notify_one(card: *mut snd_card, mask: u32, kctl: *mut snd_kcontrol, ioff: u32) {
let mut id: snd_ctl_elem_id = unsafe { (*kctl).id };
let mut lops: *mut snd_ctl_layer_ops = unsafe { core::mem::zeroed() };
let __v0: u32 = id.index;
id.index = __v0.wrapping_add(ioff);
let __v2: u32 = id.numid;
id.numid = __v2.wrapping_add(ioff);
snd_ctl_notify(card, mask, core::ptr::addr_of_mut!(id));
let mut __UNIQUE_ID_guard_522: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) });
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_522) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_4 as *mut _));
let mut __UNIQUE_ID_unlock_523: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_523) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_5 as *mut _));
lops = unsafe { snd_ctl_layer };
while !lops.is_null() {
unsafe { unsafe { (*lops).lnotify }.unwrap()(card, mask, kctl, ioff) };
lops = unsafe { (*lops).next };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_notify_one_524: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, u32, *mut snd_kcontrol, u32)>, *mut ()>(Some(snd_ctl_notify_one)) };

/// snd_ctl_new - create a new control instance with some elements
/// @kctl: the pointer to store new control instance
/// @count: the number of elements in this control
/// @access: the default access flags for elements in this control
/// @file: given when locking these elements
///
/// Allocates a memory object for a new control instance. The instance has
/// elements as many as the given number (@count). Each element has given
/// access permissions (@access). Each element is locked when @file is given.
///
/// Return: 0 on success, error code on failure
fn snd_ctl_new(kctl: &mut *mut snd_kcontrol, count: u32, access: u32, file: *mut snd_ctl_file) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
if count == 0 || count > 1028 {
return 22i32.wrapping_neg();
}
let mut __v0: *mut snd_kcontrol;
let __count: u64 = count as u64;
let __obj_size: u64 = struct_size_t__2!(__count);
let mut __obj_ptr: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__obj_ptr = __v1 as *mut snd_kcontrol;
if !__obj_ptr.is_null() {
__set_flex_counter!(__count, __obj_ptr);
}
__v0 = __obj_ptr;
unsafe {
*(kctl as *mut *mut snd_kcontrol) = __v0;
}
if unsafe { *(kctl as *mut *mut snd_kcontrol) }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*(*(kctl as *mut *mut snd_kcontrol))).count = count;
}
for idx in 0..count {
unsafe {
(*(core::ptr::addr_of_mut!((*(*(kctl as *mut *mut snd_kcontrol))).vd) as *mut snd_kcontrol_volatile).wrapping_offset(idx as isize)).access = access;
}
unsafe {
(*(core::ptr::addr_of_mut!((*(*(kctl as *mut *mut snd_kcontrol))).vd) as *mut snd_kcontrol_volatile).wrapping_offset(idx as isize)).owner = file;
}
}
0
}

/// snd_ctl_new1 - create a control instance from the template
/// @ncontrol: the initialization record
/// @private_data: the private data to set
///
/// Allocates a new struct snd_kcontrol instance and initialize from the given
/// template.  When the access field of ncontrol is 0, it's assumed as
/// READWRITE access. When the count field is 0, it's assumes as one.
///
/// Return: The pointer of the newly generated instance, or %NULL on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_new1")]
extern "C" fn snd_ctl_new1(ncontrol: *const snd_kcontrol_new, private_data: *mut ()) -> *mut snd_kcontrol {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut count: u32 = 0;
let mut access: u32 = 0;
let mut err: i32 = 0;
if snd_BUG_ON!(ncontrol.is_null() || (unsafe { (*ncontrol).info }) == None) != 0 {
return core::ptr::null_mut::<snd_kcontrol>();
}
count = unsafe { (*ncontrol).count };
if count == 0 {
count = 1;
}
access = unsafe { (*ncontrol).access };
if access == 0 {
access = 3u32;
}
access &= 503316855u32;
err = snd_ctl_new(unsafe { &mut (*core::ptr::addr_of_mut!(kctl)) }, count, access, core::ptr::null_mut::<snd_ctl_file>());
if err < 0 {
return core::ptr::null_mut::<snd_kcontrol>();
}
// The 'numid' member is decided when calling snd_ctl_add().
unsafe {
(*kctl).id.iface = (*ncontrol).iface;
}
unsafe {
(*kctl).id.device = (*ncontrol).device;
}
unsafe {
(*kctl).id.subdevice = (*ncontrol).subdevice;
}
if !unsafe { (*ncontrol).name }.is_null() {
unsafe { sized_strscpy(((*kctl).id.name.as_mut_ptr() as *mut u8) as *mut u8, (*ncontrol).name, (core::mem::size_of::<[u8; 44]>() as u64).wrapping_add(__must_be_cstr__94!() as u64).wrapping_add(__must_be_cstr__36!() as u64)) };
if (unsafe { strcmp((*ncontrol).name, ((*kctl).id.name.as_mut_ptr() as *mut u8) as *const u8) }) != 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x014ALSA: Control name '%s' truncated to '%s'\n".as_ptr() as *const u8, (*ncontrol).name, (*kctl).id.name.as_mut_ptr() as *mut u8) };
}
}
unsafe {
(*kctl).id.index = (*ncontrol).index;
}
unsafe {
(*kctl).info = (*ncontrol).info;
}
unsafe {
(*kctl).get = (*ncontrol).get;
}
unsafe {
(*kctl).put = (*ncontrol).put;
}
unsafe {
(*kctl).tlv.p = (*ncontrol).tlv.p;
}
unsafe {
(*kctl).private_value = (*ncontrol).private_value;
}
unsafe {
(*kctl).private_data = private_data;
}
kctl
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_new1_525: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const snd_kcontrol_new, *mut ()) -> *mut snd_kcontrol>, *mut ()>(Some(snd_ctl_new1)) };

/// snd_ctl_free_one - release the control instance
/// @kcontrol: the control instance
///
/// Releases the control instance created via snd_ctl_new()
/// or snd_ctl_new1().
/// Don't call this after the control was added to the card.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_free_one")]
extern "C" fn snd_ctl_free_one(kcontrol: *mut snd_kcontrol) {
if !kcontrol.is_null() {
if (unsafe { (*kcontrol).private_free }) != None {
unsafe { unsafe { (*kcontrol).private_free }.unwrap()(kcontrol) };
}
unsafe { kfree(kcontrol as *const ()) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_free_one_526: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol)>, *mut ()>(Some(snd_ctl_free_one)) };

fn snd_ctl_remove_numid_conflict(card: *mut snd_card, count: u32) -> bool {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
// Make sure that the ids assigned to the control do not wrap around
if (unsafe { (*card).last_numid }) >= 4294967295u32.wrapping_sub(count) {
unsafe {
(*card).last_numid = 0;
}
}
kctl = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*card).controls) });
while list_entry_is_head!(kctl, unsafe { core::ptr::addr_of_mut!((*card).controls) }, list) == 0 {
if (unsafe { (*kctl).id.numid }) < unsafe { (*card).last_numid }.wrapping_add(1).wrapping_add(count) && unsafe { (*kctl).id.numid }.wrapping_add(unsafe { (*kctl).count }) > unsafe { (*card).last_numid }.wrapping_add(1) {
unsafe {
(*card).last_numid = unsafe { (*kctl).id.numid }.wrapping_add((*kctl).count).wrapping_sub(1);
}
return true;
}
kctl = list_next_entry__8!(kctl, list);
}
false
}

fn snd_ctl_find_hole(card: *mut snd_card, count: u32) -> i32 {
let mut iter: u32 = 100000;
while snd_ctl_remove_numid_conflict(card, count) {
let __v1: u32 = iter.wrapping_sub(1);
iter = __v1;
if __v1 == 0 {
// this situation is very unlikely
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"unable to allocate new control numid\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
}
0
}

/// check whether the given id is contained in the given kctl
fn elem_id_matches(kctl: *const snd_kcontrol, id: *const snd_ctl_elem_id) -> bool {
let __v0: bool = (unsafe { (*kctl).id.iface }) == unsafe { (*id).iface } && (unsafe { (*kctl).id.device }) == unsafe { (*id).device } && (unsafe { (*kctl).id.subdevice }) == unsafe { (*id).subdevice } && (unsafe { strncmp(((*kctl).id.name.as_ptr() as *const u8) as *const u8, ((*id).name.as_ptr() as *const u8) as *const u8, core::mem::size_of::<[u8; 44]>() as u64) }) == 0;
__v0 && (unsafe { (*kctl).id.index }) <= unsafe { (*id).index } && unsafe { (*kctl).id.index }.wrapping_add(unsafe { (*kctl).count }) > unsafe { (*id).index }
}

// Compute a hash key for the corresponding ctl id
// It's for the name lookup, hence the numid is excluded.
// The hash key is bound in LONG_MAX to be used for Xarray key.
fn get_ctl_id_hash(id: *const snd_ctl_elem_id) -> u64 {
let mut i: i32 = 0;
let mut h: u64 = (unsafe { (*id).iface }) as u64;
h = 37u64.wrapping_mul(h).wrapping_add((unsafe { (*id).device }) as u64);
h = 37u64.wrapping_mul(h).wrapping_add((unsafe { (*id).subdevice }) as u64);
while i < 44 && (unsafe { *((*id).name.as_ptr() as *const u8).wrapping_offset(i as isize) }) != 0 {
h = 37u64.wrapping_mul(h).wrapping_add(((unsafe { *((*id).name.as_ptr() as *const u8).wrapping_offset(i as isize) }) as i32) as u64);
i = i.wrapping_add(1);
}
h = 37u64.wrapping_mul(h).wrapping_add((unsafe { (*id).index }) as u64);
let __v4: u64 = h & 9223372036854775807u64;
h = __v4;
h
}

/// add hash entries to numid and ctl xarray tables
fn add_hash_entries(card: *mut snd_card, kcontrol: *mut snd_kcontrol) {
let mut id: snd_ctl_elem_id = unsafe { (*kcontrol).id };
let mut i: i32 = 0;
unsafe { xa_store_range(core::ptr::addr_of_mut!((*card).ctl_numids), (*kcontrol).id.numid as u64, unsafe { (*kcontrol).id.numid }.wrapping_add((*kcontrol).count).wrapping_sub(1) as u64, kcontrol as *mut (), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
while (i as u32) < unsafe { (*kcontrol).count } {
id.index = unsafe { (*kcontrol).id.index }.wrapping_add(i as u32);
if xa_insert(unsafe { core::ptr::addr_of_mut!((*card).ctl_hash) }, get_ctl_id_hash(core::ptr::addr_of_mut!(id) as *const snd_ctl_elem_id), kcontrol as *mut (), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) != 0 {
// skip hash for this entry, noting we had collision
unsafe {
(*card).ctl_hash_collision = true;
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, (*card).dev as *const device, c"ctl_hash collision %d:%s:%d\n".as_ptr() as *const u8, id.iface, id.name.as_mut_ptr() as *mut u8, id.index) };
}
}
i = i.wrapping_add(1);
}
}

/// remove hash entries that have been added
fn remove_hash_entries(card: *mut snd_card, kcontrol: *mut snd_kcontrol) {
let mut id: snd_ctl_elem_id = unsafe { (*kcontrol).id };
let mut h: u64 = 0;
let mut i: i32 = 0;
while (i as u32) < unsafe { (*kcontrol).count } {
unsafe { xa_erase(core::ptr::addr_of_mut!((*card).ctl_numids), id.numid as u64) };
h = get_ctl_id_hash(core::ptr::addr_of_mut!(id) as *const snd_ctl_elem_id);
let matched: *mut snd_kcontrol = (unsafe { xa_load(core::ptr::addr_of_mut!((*card).ctl_hash), h) }) as *mut snd_kcontrol;
let __v0: bool = !matched.is_null() && (matched == kcontrol || elem_id_matches(matched as *const snd_kcontrol, core::ptr::addr_of_mut!(id) as *const snd_ctl_elem_id));
if __v0 {
unsafe { xa_erase(core::ptr::addr_of_mut!((*card).ctl_hash), h) };
}
let __v2: u32 = id.index;
id.index = __v2.wrapping_add(1);
let __v4: u32 = id.numid;
id.numid = __v4.wrapping_add(1);
i = i.wrapping_add(1);
}
}

/// add/replace a new kcontrol object; call with card->controls_rwsem locked
fn __snd_ctl_add_replace(card: *mut snd_card, kcontrol: *mut snd_kcontrol, mode: u32) -> i32 {
let mut __UNIQUE_ID_unlock_528: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_683> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut __SlateRecord_spinlock_h_683 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_528) as *mut *mut __SlateRecord_spinlock_h_683;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_683> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_spinlock_h_683 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_683;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut old: core::mem::MaybeUninit<*mut snd_kcontrol> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut snd_kcontrol = core::ptr::addr_of_mut!(old) as *mut *mut snd_kcontrol;
let mut idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(idx) as *mut u32;
let mut id: core::mem::MaybeUninit<snd_ctl_elem_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut snd_ctl_elem_id = core::ptr::addr_of_mut!(id) as *mut snd_ctl_elem_id;
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { class_write_lock_irq_destructor(__slate_cleanup_ptr_9 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { __class_write_lock_irq_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
lockdep_assert_held_write!(core::ptr::addr_of_mut!((*card).controls_rwsem));
*__slate_slot_13 = (*kcontrol).id;
if (*__slate_slot_13).index > 4294967295u32.wrapping_sub((*kcontrol).count) {
return 22i32.wrapping_neg();
} else {
*__slate_slot_11 = snd_ctl_find_id(card, core::ptr::addr_of_mut!(*__slate_slot_13) as *const snd_ctl_elem_id);
if (*__slate_slot_11).is_null() {
if (mode as u32) == 1 {
return 22i32.wrapping_neg();
}
} else {
if (mode as u32) == 0 {
while false {
}
_dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i is already present\n".as_ptr() as *const u8, (*__slate_slot_13).iface, (*__slate_slot_13).device, (*__slate_slot_13).subdevice, (*__slate_slot_13).name.as_mut_ptr() as *mut u8, (*__slate_slot_13).index);
return 16i32.wrapping_neg();
} else {
*__slate_slot_10 = snd_ctl_remove_locked(card, *__slate_slot_11);
if *__slate_slot_10 < 0 {
return *__slate_slot_10;
}
}
}
if snd_ctl_find_hole(card, (*kcontrol).count) < 0 {
return 12i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_9, class_write_lock_irq_constructor(core::ptr::addr_of_mut!((*card).controls_rwlock)));
__slate_cleanup_armed_9.set(true);
core::ptr::write(__slate_slot_8, (core::ptr::addr_of_mut!((*card).controls_rwlock) as u64) as *mut __SlateRecord_spinlock_h_683);
__slate_cleanup_armed_8.set(true);
if !class_write_lock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9))).is_null() || !class_write_lock_irq_is_conditional {
if true {
list_add_tail(core::ptr::addr_of_mut!((*kcontrol).list), core::ptr::addr_of_mut!((*card).controls));
(*card).controls_count = ((*card).controls_count as u32).wrapping_add((*kcontrol).count) as i32;
(*kcontrol).id.numid = unsafe { (*card).last_numid }.wrapping_add(1);
let __v3: u32 = (*card).last_numid;
(*card).last_numid = __v3.wrapping_add((*kcontrol).count);
}
}
__slate_cleanup_armed_8.set(false);
__class_write_lock_irq_cleanup_ctx(__slate_slot_8 as *mut _);
__slate_cleanup_armed_9.set(false);
class_write_lock_irq_destructor(__slate_slot_9 as *mut _);
add_hash_entries(card, kcontrol);
*__slate_slot_12 = 0;
while *__slate_slot_12 < (*kcontrol).count {
snd_ctl_notify_one(card, 4u32, kcontrol, *__slate_slot_12);
let __v1: u32 = *__slate_slot_12;
*__slate_slot_12 = __v1.wrapping_add(1);
}
return 0;
}
}
}
}

fn snd_ctl_add_replace(card: *mut snd_card, kcontrol: *mut snd_kcontrol, mode: u32) -> i32 {
let mut __UNIQUE_ID_unlock_530: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_530) as *mut *mut __SlateRecord_rwsem_h_270;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_270;
let mut err: i32 = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
err = 22i32.wrapping_neg();
if kcontrol.is_null() {
return err;
} else {
if snd_BUG_ON!(card.is_null() || (*kcontrol).info == None) == 0 {
core::ptr::write(__slate_slot_1, class_rwsem_write_constructor(core::ptr::addr_of_mut!((*card).controls_rwsem)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*card).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_0.set(true);
if !class_rwsem_write_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_rwsem_write_is_conditional {
if true {
err = __snd_ctl_add_replace(card, kcontrol, mode);
}
}
__slate_cleanup_armed_0.set(false);
__class_rwsem_write_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_rwsem_write_destructor(__slate_slot_1 as *mut _);
if err >= 0 {
return 0;
}
}
snd_ctl_free_one(kcontrol);
return err;
}
}
}

/// snd_ctl_add - add the control instance to the card
/// @card: the card instance
/// @kcontrol: the control instance to add
///
/// Adds the control instance created via snd_ctl_new() or
/// snd_ctl_new1() to the given card. Assigns also an unique
/// numid used for fast search.
///
/// It frees automatically the control which cannot be added.
///
/// Return: Zero if successful, or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_add")]
extern "C" fn snd_ctl_add(card: *mut snd_card, kcontrol: *mut snd_kcontrol) -> i32 {
snd_ctl_add_replace(card, kcontrol, 0u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_add_531: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_kcontrol) -> i32>, *mut ()>(Some(snd_ctl_add)) };

/// snd_ctl_replace - replace the control instance of the card
/// @card: the card instance
/// @kcontrol: the control instance to replace
/// @add_on_replace: add the control if not already added
///
/// Replaces the given control.  If the given control does not exist
/// and the add_on_replace flag is set, the control is added.  If the
/// control exists, it is destroyed first.
///
/// It frees automatically the control which cannot be added or replaced.
///
/// Return: Zero if successful, or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_replace")]
extern "C" fn snd_ctl_replace(card: *mut snd_card, kcontrol: *mut snd_kcontrol, add_on_replace: bool) -> i32 {
snd_ctl_add_replace(card, kcontrol, (if add_on_replace { 2i32 } else { 1i32 }) as u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_replace_532: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_kcontrol, bool) -> i32>, *mut ()>(Some(snd_ctl_replace)) };

fn __snd_ctl_remove(card: *mut snd_card, kcontrol: *mut snd_kcontrol, remove_hash: bool) -> i32 {
let mut __UNIQUE_ID_unlock_534: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_683> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut __SlateRecord_spinlock_h_683 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_534) as *mut *mut __SlateRecord_spinlock_h_683;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_683> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_spinlock_h_683 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_683;
let mut idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(idx) as *mut u32;
let __slate_cleanup_ptr_6 = __slate_slot_6;
let __slate_cleanup_armed_6 = core::cell::Cell::new(false);
let __slate_cleanup_6 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_6.get() {
unsafe { class_write_lock_irq_destructor(__slate_cleanup_ptr_6 as *mut _) };
}
});
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { __class_write_lock_irq_cleanup_ctx(__slate_cleanup_ptr_5 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
lockdep_assert_held_write!(core::ptr::addr_of_mut!((*card).controls_rwsem));
if snd_BUG_ON!(card.is_null() || kcontrol.is_null()) != 0 {
return 22i32.wrapping_neg();
} else {
if remove_hash {
remove_hash_entries(card, kcontrol);
}
core::ptr::write(__slate_slot_6, class_write_lock_irq_constructor(core::ptr::addr_of_mut!((*card).controls_rwlock)));
__slate_cleanup_armed_6.set(true);
core::ptr::write(__slate_slot_5, (core::ptr::addr_of_mut!((*card).controls_rwlock) as u64) as *mut __SlateRecord_spinlock_h_683);
__slate_cleanup_armed_5.set(true);
if !class_write_lock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6))).is_null() || !class_write_lock_irq_is_conditional {
if true {
list_del(core::ptr::addr_of_mut!((*kcontrol).list));
(*card).controls_count = ((*card).controls_count as u32).wrapping_sub((*kcontrol).count) as i32;
}
}
__slate_cleanup_armed_5.set(false);
__class_write_lock_irq_cleanup_ctx(__slate_slot_5 as *mut _);
__slate_cleanup_armed_6.set(false);
class_write_lock_irq_destructor(__slate_slot_6 as *mut _);
*__slate_slot_7 = 0;
while *__slate_slot_7 < (*kcontrol).count {
snd_ctl_notify_one(card, 4294967295u32, kcontrol, *__slate_slot_7);
let __v1: u32 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_add(1);
}
snd_ctl_free_one(kcontrol);
return 0;
}
}
}

fn snd_ctl_remove_locked(card: *mut snd_card, kcontrol: *mut snd_kcontrol) -> i32 {
__snd_ctl_remove(card, kcontrol, true)
}

/// snd_ctl_remove - remove the control from the card and release it
/// @card: the card instance
/// @kcontrol: the control instance to remove
///
/// Removes the control from the card and then releases the instance.
/// You don't need to call snd_ctl_free_one().
/// Passing NULL to @kcontrol argument is allowed as noop.
///
/// Return: 0 if successful, or a negative error code on failure.
///
/// Note that this function takes card->controls_rwsem lock internally.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_remove")]
extern "C" fn snd_ctl_remove(card: *mut snd_card, kcontrol: *mut snd_kcontrol) -> i32 {
if kcontrol.is_null() {
return 0;
}
let mut __UNIQUE_ID_guard_535: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_535) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_536: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_536) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
snd_ctl_remove_locked(card, kcontrol)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_remove_537: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_kcontrol) -> i32>, *mut ()>(Some(snd_ctl_remove)) };

/// snd_ctl_remove_id - remove the control of the given id and release it
/// @card: the card instance
/// @id: the control id to remove
///
/// Finds the control instance with the given id, removes it from the
/// card list and releases it.
///
/// Return: 0 if successful, or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_remove_id")]
extern "C" fn snd_ctl_remove_id(card: *mut snd_card, id: *mut snd_ctl_elem_id) -> i32 {
let mut __UNIQUE_ID_guard_538: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_538) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_539: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_539) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let kctl: *mut snd_kcontrol = snd_ctl_find_id(card, id as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
snd_ctl_remove_locked(card, kctl)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_remove_id_540: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_elem_id) -> i32>, *mut ()>(Some(snd_ctl_remove_id)) };

/// snd_ctl_remove_user_ctl - remove and release the unlocked user control
/// @file: active control handle
/// @id: the control id to remove
///
/// Finds the control instance with the given id, removes it from the
/// card list and releases it.
///
/// Return: 0 if successful, or a negative error code on failure.
fn snd_ctl_remove_user_ctl(file: *mut snd_ctl_file, id: *mut snd_ctl_elem_id) -> i32 {
let card: *mut snd_card = unsafe { (*file).card };
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut idx: i32 = 0;
let mut __UNIQUE_ID_guard_541: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_541) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_542: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_542) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
kctl = snd_ctl_find_id(card, id as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
if (unsafe { (*(core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile).wrapping_offset(0 as isize)).access }) & 536870912u32 == 0 {
return 22i32.wrapping_neg();
}
while (idx as u32) < unsafe { (*kctl).count } {
if !unsafe { (*(core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile).wrapping_offset(idx as isize)).owner }.is_null() && (unsafe { (*(core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile).wrapping_offset(idx as isize)).owner }) != file {
return 16i32.wrapping_neg();
}
idx = idx.wrapping_add(1);
}
snd_ctl_remove_locked(card, kctl)
}

/// snd_ctl_activate_id - activate/inactivate the control of the given id
/// @card: the card instance
/// @id: the control id to activate/inactivate
/// @active: non-zero to activate
///
/// Finds the control instance with the given id, and activate or
/// inactivate the control together with notification, if changed.
/// The given ID data is filled with full information.
///
/// Return: 0 if unchanged, 1 if changed, or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_activate_id")]
extern "C" fn snd_ctl_activate_id(card: *mut snd_card, id: *mut snd_ctl_elem_id, active: i32) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut index_offset: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(index_offset) as *mut u32;
let mut vd: core::mem::MaybeUninit<*mut snd_kcontrol_volatile> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut snd_kcontrol_volatile = core::ptr::addr_of_mut!(vd) as *mut *mut snd_kcontrol_volatile;
let mut kctl: core::mem::MaybeUninit<*mut snd_kcontrol> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut snd_kcontrol = core::ptr::addr_of_mut!(kctl) as *mut *mut snd_kcontrol;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
down_write(core::ptr::addr_of_mut!((*card).controls_rwsem));
*__slate_slot_9 = snd_ctl_find_id(card, id as *const snd_ctl_elem_id);
if (*__slate_slot_9).is_null() {
*__slate_slot_6 = 2i32.wrapping_neg();
} else {
*__slate_slot_7 = snd_ctl_get_ioff(*__slate_slot_9, id);
*__slate_slot_8 = (core::ptr::addr_of_mut!((*(*__slate_slot_9)).vd) as *mut snd_kcontrol_volatile).wrapping_offset(*__slate_slot_7 as isize);
*__slate_slot_6 = 0;
if active != 0 {
if (*(*__slate_slot_8)).access & 256u32 == 0 {
break '__join_0;
} else {
let __v2: *mut snd_kcontrol_volatile = *__slate_slot_8;
let __v0: u32 = (*__v2).access & 4294967039u32;
(*__v2).access = __v0;
}
} else {
if (*(*__slate_slot_8)).access & 256u32 != 0 {
break '__join_0;
} else {
let __v5: *mut snd_kcontrol_volatile = *__slate_slot_8;
(*__v5).access = (*__v5).access | 256u32;
}
}
snd_ctl_build_ioff(id, *__slate_slot_9, *__slate_slot_7);
downgrade_write(core::ptr::addr_of_mut!((*card).controls_rwsem));
snd_ctl_notify_one(card, 2u32, *__slate_slot_9, *__slate_slot_7);
up_read(core::ptr::addr_of_mut!((*card).controls_rwsem));
return 1;
}
}
up_write(core::ptr::addr_of_mut!((*card).controls_rwsem));
return *__slate_slot_6;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_activate_id_543: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_elem_id, i32) -> i32>, *mut ()>(Some(snd_ctl_activate_id)) };

/// snd_ctl_rename_id - replace the id of a control on the card
/// @card: the card instance
/// @src_id: the old id
/// @dst_id: the new id
///
/// Finds the control with the old id from the card, and replaces the
/// id with the new one.
///
/// The function tries to keep the already assigned numid while replacing
/// the rest.
///
/// Note that this function should be used only in the card initialization
/// phase.  Calling after the card instantiation may cause issues with
/// user-space expecting persistent numids.
///
/// Return: Zero if successful, or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_rename_id")]
extern "C" fn snd_ctl_rename_id(card: *mut snd_card, src_id: *mut snd_ctl_elem_id, dst_id: *mut snd_ctl_elem_id) -> i32 {
let mut saved_numid: i32 = 0;
let mut __UNIQUE_ID_guard_544: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_544) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_545: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_545) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let kctl: *mut snd_kcontrol = snd_ctl_find_id(card, src_id as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
saved_numid = (unsafe { (*kctl).id.numid }) as i32;
remove_hash_entries(card, kctl);
unsafe {
(*kctl).id = *dst_id;
}
unsafe {
(*kctl).id.numid = saved_numid as u32;
}
add_hash_entries(card, kctl);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_rename_id_546: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_elem_id, *mut snd_ctl_elem_id) -> i32>, *mut ()>(Some(snd_ctl_rename_id)) };

/// snd_ctl_rename - rename the control on the card
/// @card: the card instance
/// @kctl: the control to rename
/// @name: the new name
///
/// Renames the specified control on the card to the new name.
///
/// Note that this function takes card->controls_rwsem lock internally.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_rename")]
extern "C" fn snd_ctl_rename(card: *mut snd_card, kctl: *mut snd_kcontrol, name: *const u8) {
let mut __UNIQUE_ID_guard_547: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_547) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_548: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_548) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
remove_hash_entries(card, kctl);
if (unsafe { sized_strscpy(((*kctl).id.name.as_mut_ptr() as *mut u8) as *mut u8, name, (core::mem::size_of::<[u8; 44]>() as u64).wrapping_add(__must_be_cstr__54!() as u64).wrapping_add(__must_be_cstr__41!() as u64)) }) < 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014ALSA: Renamed control new name '%s' truncated to '%s'\n".as_ptr() as *const u8, name, (*kctl).id.name.as_mut_ptr() as *mut u8) };
}
add_hash_entries(card, kctl);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_rename_549: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *mut snd_kcontrol, *const u8)>, *mut ()>(Some(snd_ctl_rename)) };

/// snd_ctl_find_numid - find the control instance with the given number-id
/// @card: the card instance
/// @numid: the number-id to search
///
/// Finds the control instance with the given number-id from the card.
///
/// Return: The pointer of the instance if found, or %NULL if not.
///
/// Note that this function takes card->controls_rwlock lock internally.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_find_numid")]
extern "C" fn snd_ctl_find_numid(card: *mut snd_card, numid: u32) -> *mut snd_kcontrol {
if snd_BUG_ON!(card.is_null() || numid == 0) != 0 {
return core::ptr::null_mut::<snd_kcontrol>();
}
(unsafe { xa_load(core::ptr::addr_of_mut!((*card).ctl_numids), numid as u64) }) as *mut snd_kcontrol
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_find_numid_550: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, u32) -> *mut snd_kcontrol>, *mut ()>(Some(snd_ctl_find_numid)) };

/// snd_ctl_find_id - find the control instance with the given id
/// @card: the card instance
/// @id: the id to search
///
/// Finds the control instance with the given id from the card.
///
/// Return: The pointer of the instance if found, or %NULL if not.
///
/// Note that this function takes card->controls_rwlock lock internally.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_find_id")]
extern "C" fn snd_ctl_find_id(card: *mut snd_card, id: *const snd_ctl_elem_id) -> *mut snd_kcontrol {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
if snd_BUG_ON!(card.is_null() || id.is_null()) != 0 {
return core::ptr::null_mut::<snd_kcontrol>();
}
if (unsafe { (*id).numid }) != 0 {
return snd_ctl_find_numid(card, unsafe { (*id).numid });
}
kctl = (unsafe { xa_load(core::ptr::addr_of_mut!((*card).ctl_hash), get_ctl_id_hash(id)) }) as *mut snd_kcontrol;
let __v2: bool = !kctl.is_null() && elem_id_matches(kctl as *const snd_kcontrol, id);
if __v2 {
return kctl;
}
if !unsafe { (*card).ctl_hash_collision } {
return core::ptr::null_mut::<snd_kcontrol>();
}
// we can rely on only hash table
// no matching in hash table - try all as the last resort
let mut __UNIQUE_ID_guard_551: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwlock) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_551) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_552: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwlock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_552) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
kctl = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*card).controls) });
while list_entry_is_head!(kctl, unsafe { core::ptr::addr_of_mut!((*card).controls) }, list) == 0 {
if elem_id_matches(kctl as *const snd_kcontrol, id) {
return kctl;
}
kctl = list_next_entry__8!(kctl, list);
}
core::ptr::null_mut::<snd_kcontrol>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_find_id_553: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, *const snd_ctl_elem_id) -> *mut snd_kcontrol>, *mut ()>(Some(snd_ctl_find_id)) };

fn snd_ctl_card_info(card: *mut snd_card, ctl: *mut snd_ctl_file, cmd: u32, arg: *mut ()) -> i32 {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut components: *const u8 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_555: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_555) as *mut *mut __SlateRecord_rwsem_h_259;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_259;
let mut n: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i64 = core::ptr::addr_of_mut!(n) as *mut i64;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut info: core::mem::MaybeUninit<*mut snd_ctl_card_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut snd_ctl_card_info = core::ptr::addr_of_mut!(info) as *mut *mut snd_ctl_card_info;
let __slate_cleanup_ptr_10 = __slate_slot_10;
let __slate_cleanup_armed_10 = core::cell::Cell::new(false);
let __slate_cleanup_10 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_10.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_10 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _) };
}
});
unsafe {
__slate_cleanup_armed_10.set(true);
__obj_size = size_mul(core::mem::size_of::<snd_ctl_card_info>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
}
*__slate_slot_10 = *__slate_slot_5 as *mut snd_ctl_card_info;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if (*__slate_slot_10).is_null() {
return 12i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_2, class_rwsem_read_constructor(core::ptr::addr_of_mut!(snd_ioctl_rwsem)));
__slate_cleanup_armed_2.set(true);
core::ptr::write(__slate_slot_1, (core::ptr::addr_of_mut!(snd_ioctl_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_1.set(true);
if !class_rwsem_read_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2))).is_null() || !class_rwsem_read_is_conditional {
if true {
components = (*card).components as *const u8;
if components.is_null() {
components = c"".as_ptr() as *const u8;
}
(*(*__slate_slot_10)).card = (*card).number;
sized_strscpy(((*(*__slate_slot_10)).id.as_mut_ptr() as *mut u8) as *mut u8, ((*card).id.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 16]>() as u64).wrapping_add(__must_be_cstr__98!() as u64).wrapping_add(__must_be_cstr__66!() as u64));
sized_strscpy(((*(*__slate_slot_10)).driver.as_mut_ptr() as *mut u8) as *mut u8, ((*card).driver.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 16]>() as u64).wrapping_add(__must_be_cstr__103!() as u64).wrapping_add(__must_be_cstr__64!() as u64));
sized_strscpy(((*(*__slate_slot_10)).name.as_mut_ptr() as *mut u8) as *mut u8, ((*card).shortname.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 32]>() as u64).wrapping_add(__must_be_cstr__74!() as u64).wrapping_add(__must_be_cstr__27!() as u64));
sized_strscpy(((*(*__slate_slot_10)).longname.as_mut_ptr() as *mut u8) as *mut u8, ((*card).longname.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 80]>() as u64).wrapping_add(__must_be_cstr__81!() as u64).wrapping_add(__must_be_cstr__87!() as u64));
sized_strscpy(((*(*__slate_slot_10)).mixername.as_mut_ptr() as *mut u8) as *mut u8, ((*card).mixername.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 80]>() as u64).wrapping_add(__must_be_cstr__29!() as u64).wrapping_add(__must_be_cstr__88!() as u64));
*__slate_slot_3 = sized_strscpy(((*(*__slate_slot_10)).components.as_mut_ptr() as *mut u8) as *mut u8, components, (core::mem::size_of::<[u8; 128]>() as u64).wrapping_add(__must_be_cstr__25!() as u64).wrapping_add(__must_be_cstr__97!() as u64));
if *__slate_slot_3 < 0 {
*((*(*__slate_slot_10)).components.as_mut_ptr() as *mut u8).wrapping_offset((core::mem::size_of::<[u8; 128]>() as u64).wrapping_sub(2) as isize) = 62;
}
// mark the truncation with '>' before NULL terminator
}
}
__slate_cleanup_armed_1.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_1 as *mut _);
__slate_cleanup_armed_2.set(false);
class_rwsem_read_destructor(__slate_slot_2 as *mut _);
if copy_to_user(arg, *__slate_slot_10 as *const (), core::mem::size_of::<snd_ctl_card_info>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
}
}
}

fn snd_ctl_card_bytes(card: *mut snd_card, info: *mut snd_ctl_card_bytes, data_len_out: *mut u32) -> i32 {
let mut __x: u32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(__val_pu) as *mut u32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut components: *const u8 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_557: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_557) as *mut *mut __SlateRecord_rwsem_h_259;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_259;
let mut data_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(data_len) as *mut u32;
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_9 as *mut _) };
}
});
let __slate_cleanup_ptr_8 = __slate_slot_8;
let __slate_cleanup_armed_8 = core::cell::Cell::new(false);
let __slate_cleanup_8 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_8.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __t1: u32 = (*info).r#type;
if __t1 == 1 {
core::ptr::write(__slate_slot_9, class_rwsem_read_constructor(core::ptr::addr_of_mut!(snd_ioctl_rwsem)));
__slate_cleanup_armed_9.set(true);
core::ptr::write(__slate_slot_8, (core::ptr::addr_of_mut!(snd_ioctl_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_8.set(true);
if !class_rwsem_read_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_9))).is_null() || !class_rwsem_read_is_conditional {
if true {
components = (*card).components as *const u8;
if components.is_null() {
components = c"".as_ptr() as *const u8;
}
*__slate_slot_10 = strlen(components).wrapping_add(1) as u32;
if !((*info).data == 0 || (*info).data_allocated == 0) {
if (*info).data_allocated < *__slate_slot_10 {
return 12i32.wrapping_neg();
} else {
if copy_to_user(u64_to_user_ptr!(unsafe { (*info).data }), components as *const (), *__slate_slot_10 as u64) != 0 {
return 14i32.wrapping_neg();
}
}
}
}
}
__slate_cleanup_armed_8.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_8 as *mut _);
__slate_cleanup_armed_9.set(false);
class_rwsem_read_destructor(__slate_slot_9 as *mut _);
might_fault();
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x = *__slate_slot_10;
let __ptr: *mut u32 = data_len_out;
__chk_user_ptr!();
*__slate_slot_3 = __ptr as *mut ();
*__slate_slot_2 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_3 as u64 => __t0, in("ax") *__slate_slot_2 as u32, options(att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user__2!(__x, __ptr);
if (*__slate_slot_4 as i64) != 0 {
return 14i32.wrapping_neg();
} else {
return 0;
}
} else {
return 22i32.wrapping_neg();
}
}
}

fn snd_ctl_card_bytes_user(card: *mut snd_card, _info: *mut snd_ctl_card_bytes) -> i32 {
let mut info: snd_ctl_card_bytes = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
if copy_from_user(core::ptr::addr_of_mut!(info) as *mut (), _info as *const (), core::mem::size_of::<snd_ctl_card_bytes>() as u64) != 0 {
return 14i32.wrapping_neg();
}
snd_ctl_card_bytes(card, core::ptr::addr_of_mut!(info), unsafe { core::ptr::addr_of_mut!((*_info).data_len) })
}

fn snd_ctl_elem_list(card: *mut snd_card, list: &mut snd_ctl_elem_list) -> i32 {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut offset: u32 = unsafe { (*(list as *mut snd_ctl_elem_list)).offset };
let mut space: u32 = unsafe { (*(list as *mut snd_ctl_elem_list)).space };
let mut jidx: u32 = 0;
let mut __UNIQUE_ID_guard_558: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_558) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_559: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_559) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
(*(list as *mut snd_ctl_elem_list)).count = (*card).controls_count as u32;
}
unsafe {
(*(list as *mut snd_ctl_elem_list)).used = 0;
}
if space == 0 {
return 0;
}
kctl = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*card).controls) });
while list_entry_is_head!(kctl, unsafe { core::ptr::addr_of_mut!((*card).controls) }, list) == 0 {
if offset >= unsafe { (*kctl).count } {
offset = offset.wrapping_sub(unsafe { (*kctl).count });
} else {
jidx = offset;
while jidx < unsafe { (*kctl).count } {
snd_ctl_build_ioff(core::ptr::addr_of_mut!(id), kctl, jidx);
if copy_to_user(unsafe { (*(list as *mut snd_ctl_elem_list)).pids }.wrapping_offset((unsafe { (*(list as *mut snd_ctl_elem_list)).used }) as isize) as *mut (), core::ptr::addr_of_mut!(id) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
let __v7: u32 = unsafe { (*list).used };
unsafe {
(*list).used = __v7.wrapping_add(1);
}
let __v10: u32 = space.wrapping_sub(1);
space = __v10;
if __v10 == 0 {
return 0;
}
jidx = jidx.wrapping_add(1);
}
offset = 0;
}
kctl = list_next_entry__8!(kctl, list);
}
0
}

fn snd_ctl_elem_list_user(card: *mut snd_card, _list: *mut snd_ctl_elem_list) -> i32 {
let mut list: snd_ctl_elem_list = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(list) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 80usize);
}
let mut err: i32 = 0;
if copy_from_user(core::ptr::addr_of_mut!(list) as *mut (), _list as *const (), core::mem::size_of::<snd_ctl_elem_list>() as u64) != 0 {
return 14i32.wrapping_neg();
}
err = snd_ctl_elem_list(card, unsafe { &mut (*core::ptr::addr_of_mut!(list)) });
if err != 0 {
return err;
}
if copy_to_user(_list as *mut (), core::ptr::addr_of_mut!(list) as *const (), core::mem::size_of::<snd_ctl_elem_list>() as u64) != 0 {
return 14i32.wrapping_neg();
}
0
}

/// Check whether the given kctl info is valid
fn snd_ctl_check_elem_info(card: *mut snd_card, info: *const snd_ctl_elem_info) -> i32 {
if (unsafe { (*info).r#type }) < 1 || (unsafe { (*info).r#type }) > 6 {
if !card.is_null() {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i: invalid type %d\n".as_ptr() as *const u8, (*info).id.iface, (*info).id.device, (*info).id.subdevice, (*info).id.name.as_ptr() as *const u8, (*info).id.index, (*info).r#type) };
}
return 22i32.wrapping_neg();
}
if (unsafe { (*info).r#type }) == 3 && (unsafe { (*info).value.enumerated.items }) == 0 {
if !card.is_null() {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i: zero enum items\n".as_ptr() as *const u8, (*info).id.iface, (*info).id.device, (*info).id.subdevice, (*info).id.name.as_ptr() as *const u8, (*info).id.index) };
}
return 22i32.wrapping_neg();
}
if (unsafe { (*info).count }) > unsafe { max_value_counts.0[(*info).r#type as usize] } {
if !card.is_null() {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i: invalid count %d\n".as_ptr() as *const u8, (*info).id.iface, (*info).id.device, (*info).id.subdevice, (*info).id.name.as_ptr() as *const u8, (*info).id.index, (*info).count) };
}
return 22i32.wrapping_neg();
}
0
}

static max_value_counts: __SlateAlign16<[u32; 7]> = __SlateAlign16({
let mut __t0: [u32; 7] = unsafe { core::mem::zeroed() };
__t0[1] = 128u32;
__t0[2] = 128u32;
__t0[3] = 128u32;
__t0[4] = 512u32;
__t0[5] = 1u32;
__t0[6] = 64u32;
__t0
});

/// The capacity of struct snd_ctl_elem_value.value.
static value_sizes: __SlateAlign16<[u32; 7]> = __SlateAlign16({
let mut __t0: [u32; 7] = unsafe { core::mem::zeroed() };
__t0[1] = (core::mem::size_of::<i64>() as u64) as u32;
__t0[2] = (core::mem::size_of::<i64>() as u64) as u32;
__t0[3] = (core::mem::size_of::<u32>() as u64) as u32;
__t0[4] = (core::mem::size_of::<u8>() as u64) as u32;
__t0[5] = (core::mem::size_of::<snd_aes_iec958>() as u64) as u32;
__t0[6] = (core::mem::size_of::<i64>() as u64) as u32;
__t0
});

/// fill the remaining snd_ctl_elem_value data with the given pattern
fn fill_remaining_elem_value(control: *mut snd_ctl_elem_value, info: &snd_ctl_elem_info, pattern: u32) {
let mut offset: u64 = unsafe { value_sizes.0[(*(info as *const snd_ctl_elem_info)).r#type as usize] }.wrapping_mul(unsafe { (*(info as *const snd_ctl_elem_info)).count }) as u64;
offset = offset.wrapping_add(core::mem::size_of::<u32>() as u64).wrapping_sub(1) / (core::mem::size_of::<u32>() as u64);
memset32(((unsafe { (*control).value.bytes.data.as_mut_ptr() as *mut u8 }) as *mut u32).wrapping_offset(offset as isize), pattern, ((core::mem::size_of::<__SlateRecord_asound_h_1190>() as u64) / (core::mem::size_of::<u32>() as u64)).wrapping_sub(offset));
}

/// check whether the given integer ctl value is valid
fn sanity_check_int_value(card: *mut snd_card, control: *const snd_ctl_elem_value, info: *const snd_ctl_elem_info, i: i32, print_error: bool) -> i32 {
let mut lval: i64 = 0;
let mut lmin: i64 = 0;
let mut lmax: i64 = 0;
let mut lstep: i64 = 0;
let mut rem: u64 = 0;
match unsafe { (*info).r#type } {
2 => {
lval = unsafe { *((*control).value.integer.value.as_ptr() as *const i64).wrapping_offset(i as isize) };
lmin = unsafe { (*info).value.integer.min };
lmax = unsafe { (*info).value.integer.max };
lstep = unsafe { (*info).value.integer.step };
}
6 => {
lval = unsafe { *((*control).value.integer64.value.as_ptr() as *const i64).wrapping_offset(i as isize) };
lmin = unsafe { (*info).value.integer64.min };
lmax = unsafe { (*info).value.integer64.max };
lstep = unsafe { (*info).value.integer64.step };
}
3 => {
lval = (unsafe { *((*control).value.enumerated.item.as_ptr() as *const u32).wrapping_offset(i as isize) }) as i64;
lmin = 0;
lmax = unsafe { (*info).value.enumerated.items }.wrapping_sub(1) as i64;
lstep = 0;
}
_ => {
lval = unsafe { *((*control).value.integer.value.as_ptr() as *const i64).wrapping_offset(i as isize) };
lmin = 0;
lmax = 1;
lstep = 0;
}
}
if lval < lmin || lval > lmax {
if print_error {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i: value out of range %lld (%lld/%lld) at count %i\n".as_ptr() as *const u8, (*control).id.iface, (*control).id.device, (*control).id.subdevice, (*control).id.name.as_ptr() as *const u8, (*control).id.index, lval, lmin, lmax, i) };
}
return 22i32.wrapping_neg();
}
if lstep != 0 {
div64_u64_rem(lval as u64, lstep as u64, unsafe { &mut (*core::ptr::addr_of_mut!(rem)) });
if rem != 0 {
if print_error {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i: unaligned value %lld (step %lld) at count %i\n".as_ptr() as *const u8, (*control).id.iface, (*control).id.device, (*control).id.subdevice, (*control).id.name.as_ptr() as *const u8, (*control).id.index, lval, lstep, i) };
}
return 22i32.wrapping_neg();
}
}
0
}

/// check whether the all input values are valid for the given elem value
fn sanity_check_input_values(card: *mut snd_card, control: *const snd_ctl_elem_value, info: *const snd_ctl_elem_info, print_error: bool) -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
match unsafe { (*info).r#type } {
1..=3 | 6 => {
i = 0;
while (i as u32) < unsafe { (*info).count } {
ret = sanity_check_int_value(card, control, info, i, print_error);
if ret < 0 {
return ret;
}
i = i.wrapping_add(1);
}
}
_ => {
}
}
0
}

/// perform sanity checks to the given snd_ctl_elem_value object
fn sanity_check_elem_value(card: *mut snd_card, control: *const snd_ctl_elem_value, info: &snd_ctl_elem_info, pattern: u32) -> i32 {
let mut offset: u64 = 0;
let mut ret: i32 = 0;
let mut p: *mut u32 = unsafe { core::mem::zeroed() };
ret = sanity_check_input_values(card, control, info as *const snd_ctl_elem_info, true);
if ret < 0 {
return ret;
}
// check whether the remaining area kept untouched
offset = unsafe { value_sizes.0[(*(info as *const snd_ctl_elem_info)).r#type as usize] }.wrapping_mul(unsafe { (*(info as *const snd_ctl_elem_info)).count }) as u64;
offset = offset.wrapping_add(core::mem::size_of::<u32>() as u64).wrapping_sub(1) / (core::mem::size_of::<u32>() as u64);
p = ((unsafe { (*control).value.bytes.data.as_ptr() as *const u8 }) as *mut u32).wrapping_offset(offset as isize);
while offset < (core::mem::size_of::<__SlateRecord_asound_h_1190>() as u64) / (core::mem::size_of::<u32>() as u64) {
if (unsafe { *p }) != pattern {
ret = 22i32.wrapping_neg();
break;
}
unsafe {
*p = 0;
}
// clear the checked area
offset = offset.wrapping_add(1);
let __v3: *mut u32 = p;
p = __v3.wrapping_offset(1 as isize);
}
ret
}

fn __snd_ctl_elem_info(card: *mut snd_card, kctl: *mut snd_kcontrol, info: *mut snd_ctl_elem_info, ctl: *mut snd_ctl_file) -> i32 {
let mut index_offset: u32 = 0;
let mut result: i32 = unsafe { unsafe { (*kctl).info }.unwrap()(kctl, info) };
if result >= 0 {
snd_BUG_ON__5!(unsafe { (*info).access });
index_offset = snd_ctl_get_ioff(kctl, unsafe { core::ptr::addr_of_mut!((*info).id) });
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(index_offset as isize);
snd_ctl_build_ioff(unsafe { core::ptr::addr_of_mut!((*info).id) }, kctl, index_offset);
unsafe {
(*info).access = (*vd).access;
}
if !unsafe { (*vd).owner }.is_null() {
unsafe {
(*info).access = (*info).access | 512u32;
}
if (unsafe { (*vd).owner }) == ctl {
unsafe {
(*info).access = (*info).access | 1024u32;
}
}
unsafe {
(*info).owner = pid_vnr((*(*vd).owner).pid);
}
} else {
unsafe {
(*info).owner = 1i32.wrapping_neg();
}
}
let __v7: bool = snd_ctl_skip_validation!() == 0 && snd_ctl_check_elem_info(card, info as *const snd_ctl_elem_info) < 0;
if __v7 {
result = 22i32.wrapping_neg();
}
}
result
}

fn snd_ctl_elem_info(ctl: *mut snd_ctl_file, info: *mut snd_ctl_elem_info) -> i32 {
let card: *mut snd_card = unsafe { (*ctl).card };
let mut __UNIQUE_ID_guard_560: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_560) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_561: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_561) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let kctl: *mut snd_kcontrol = snd_ctl_find_id(card, (unsafe { core::ptr::addr_of_mut!((*info).id) }) as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
__snd_ctl_elem_info(card, kctl, info, ctl)
}

fn snd_ctl_elem_info_user(ctl: *mut snd_ctl_file, _info: *mut snd_ctl_elem_info) -> i32 {
let card: *mut snd_card = unsafe { (*ctl).card };
let mut info: snd_ctl_elem_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 272usize);
}
let mut result: i32 = 0;
if copy_from_user(core::ptr::addr_of_mut!(info) as *mut (), _info as *const (), core::mem::size_of::<snd_ctl_elem_info>() as u64) != 0 {
return 14i32.wrapping_neg();
}
result = unsafe { snd_power_ref_and_wait(card) };
if result != 0 {
return result;
}
result = snd_ctl_elem_info(ctl, core::ptr::addr_of_mut!(info));
snd_power_unref(card);
if result < 0 {
return result;
}
// drop internal access flags
let __v1: u32 = info.access & 4060086271u32;
info.access = __v1;
if copy_to_user(_info as *mut (), core::ptr::addr_of_mut!(info) as *const (), core::mem::size_of::<snd_ctl_elem_info>() as u64) != 0 {
return 14i32.wrapping_neg();
}
result
}

fn snd_ctl_elem_read(card: *mut snd_card, control: *mut snd_ctl_elem_value) -> i32 {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut index_offset: u32 = 0;
let mut info: snd_ctl_elem_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 272usize);
}
let pattern: u32 = 3735928559;
let mut ret: i32 = 0;
let mut __UNIQUE_ID_guard_562: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_562) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_563: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_563) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
kctl = snd_ctl_find_id(card, (unsafe { core::ptr::addr_of_mut!((*control).id) }) as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
index_offset = snd_ctl_get_ioff(kctl, unsafe { core::ptr::addr_of_mut!((*control).id) });
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(index_offset as isize);
if (unsafe { (*vd).access }) & 1u32 == 0 || (unsafe { (*kctl).get }) == None {
return 1i32.wrapping_neg();
}
snd_ctl_build_ioff(unsafe { core::ptr::addr_of_mut!((*control).id) }, kctl, index_offset);
if snd_ctl_skip_validation!() == 0 {
fill_remaining_elem_value(control, unsafe { &(*core::ptr::addr_of_mut!(info)) }, pattern);
}
ret = unsafe { unsafe { (*kctl).get }.unwrap()(kctl, control) };
if ret < 0 {
return ret;
}
let __v2: bool = snd_ctl_skip_validation!() == 0 && sanity_check_elem_value(card, control as *const snd_ctl_elem_value, unsafe { &(*(core::ptr::addr_of_mut!(info) as *const snd_ctl_elem_info)) }, pattern) < 0;
if __v2 {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"control %i:%i:%i:%s:%i: access overflow\n".as_ptr() as *const u8, (*control).id.iface, (*control).id.device, (*control).id.subdevice, (*control).id.name.as_mut_ptr() as *mut u8, (*control).id.index) };
return 22i32.wrapping_neg();
}
0
}

fn snd_ctl_elem_read_user(card: *mut snd_card, _control: *mut snd_ctl_elem_value) -> i32 {
let mut result: i32 = 0;
let mut control: *mut snd_ctl_elem_value = (unsafe { memdup_user(_control as *const (), core::mem::size_of::<snd_ctl_elem_value>() as u64) }) as *mut snd_ctl_elem_value;
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(control) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
if IS_ERR(control as *const ()) {
return PTR_ERR(control as *const ()) as i32;
}
result = unsafe { snd_power_ref_and_wait(card) };
if result != 0 {
return result;
}
result = snd_ctl_elem_read(card, control);
snd_power_unref(card);
if result < 0 {
return result;
}
if copy_to_user(_control as *mut (), control as *const (), core::mem::size_of::<snd_ctl_elem_value>() as u64) != 0 {
return 14i32.wrapping_neg();
}
result
}

fn snd_ctl_put(card: *mut snd_card, kctl: *mut snd_kcontrol, control: *mut snd_ctl_elem_value, access: u32) -> i32 {
unsafe { unsafe { (*kctl).put }.unwrap()(kctl, control) }
}

fn snd_ctl_elem_write(card: *mut snd_card, file: *mut snd_ctl_file, control: *mut snd_ctl_elem_value) -> i32 {
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut index_offset: u32 = 0;
let mut result: i32 = 0;
unsafe { down_write(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
kctl = snd_ctl_find_id(card, (unsafe { core::ptr::addr_of_mut!((*control).id) }) as *const snd_ctl_elem_id);
if kctl.is_null() {
unsafe { up_write(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
return 2i32.wrapping_neg();
}
index_offset = snd_ctl_get_ioff(kctl, unsafe { core::ptr::addr_of_mut!((*control).id) });
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(index_offset as isize);
if (unsafe { (*vd).access }) & 2u32 == 0 || (unsafe { (*kctl).put }) == None || !file.is_null() && !unsafe { (*vd).owner }.is_null() && (unsafe { (*vd).owner }) != file {
unsafe { up_write(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
return 1i32.wrapping_neg();
}
snd_ctl_build_ioff(unsafe { core::ptr::addr_of_mut!((*control).id) }, kctl, index_offset);
// validate input values
if IS_ENABLED!() != 0 {
let mut info: snd_ctl_elem_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 272usize);
}
unsafe { memset(core::ptr::addr_of_mut!(info) as *mut (), 0i32, core::mem::size_of::<snd_ctl_elem_info>() as u64) };
info.id = unsafe { (*control).id };
result = __snd_ctl_elem_info(card, kctl, core::ptr::addr_of_mut!(info), core::ptr::null_mut::<snd_ctl_file>());
if result == 0 {
result = sanity_check_input_values(card, control as *const snd_ctl_elem_value, core::ptr::addr_of_mut!(info) as *const snd_ctl_elem_info, false);
}
}
if result == 0 {
result = snd_ctl_put(card, kctl, control, unsafe { (*vd).access });
}
if result < 0 {
unsafe { up_write(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
return result;
}
if result > 0 {
unsafe { downgrade_write(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
snd_ctl_notify_one(card, 1u32, kctl, index_offset);
unsafe { up_read(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
} else {
unsafe { up_write(core::ptr::addr_of_mut!((*card).controls_rwsem)) };
}
0
}

fn snd_ctl_elem_write_user(file: *mut snd_ctl_file, _control: *mut snd_ctl_elem_value) -> i32 {
let mut result: i32 = 0;
let mut control: *mut snd_ctl_elem_value = (unsafe { memdup_user(_control as *const (), core::mem::size_of::<snd_ctl_elem_value>() as u64) }) as *mut snd_ctl_elem_value;
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(control) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
if IS_ERR(control as *const ()) {
return PTR_ERR(control as *const ()) as i32;
}
let card: *mut snd_card = unsafe { (*file).card };
result = unsafe { snd_power_ref_and_wait(card) };
if result < 0 {
return result;
}
result = snd_ctl_elem_write(card, file, control);
snd_power_unref(card);
if result < 0 {
return result;
}
if copy_to_user(_control as *mut (), control as *const (), core::mem::size_of::<snd_ctl_elem_value>() as u64) != 0 {
return 14i32.wrapping_neg();
}
result
}

fn snd_ctl_elem_lock(file: *mut snd_ctl_file, _id: *mut snd_ctl_elem_id) -> i32 {
let card: *mut snd_card = unsafe { (*file).card };
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
if copy_from_user(core::ptr::addr_of_mut!(id) as *mut (), _id as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_564: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_564) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_565: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_565) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
kctl = snd_ctl_find_id(card, core::ptr::addr_of_mut!(id) as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(snd_ctl_get_ioff(kctl, core::ptr::addr_of_mut!(id)) as isize);
if !unsafe { (*vd).owner }.is_null() {
return 16i32.wrapping_neg();
}
unsafe {
(*vd).owner = file;
}
0
}

fn snd_ctl_elem_unlock(file: *mut snd_ctl_file, _id: *mut snd_ctl_elem_id) -> i32 {
let card: *mut snd_card = unsafe { (*file).card };
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
if copy_from_user(core::ptr::addr_of_mut!(id) as *mut (), _id as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_566: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_566) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_567: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_567) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
kctl = snd_ctl_find_id(card, core::ptr::addr_of_mut!(id) as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(snd_ctl_get_ioff(kctl, core::ptr::addr_of_mut!(id)) as isize);
if unsafe { (*vd).owner }.is_null() {
return 22i32.wrapping_neg();
}
if (unsafe { (*vd).owner }) != file {
return 1i32.wrapping_neg();
}
unsafe {
(*vd).owner = core::ptr::null_mut::<snd_ctl_file>();
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct user_element {
    info: snd_ctl_elem_info,
    card: *mut snd_card,
    /// element data
    elem_data: *mut u8,
    /// size of element data in bytes
    elem_data_size: u64,
    /// TLV data
    tlv_data: *mut (),
    /// TLV data size
    tlv_data_size: u64,
    /// private data (like strings for enumerated type)
    priv_data: *mut (),
}


/// check whether the addition (in bytes) of user ctl element may overflow the limit.
fn check_user_elem_overflow(card: *mut snd_card, add: i64) -> bool {
((unsafe { (*card).user_ctl_alloc_size }) as i64).wrapping_add(add) > ((unsafe { max_user_ctl_alloc_size }) as i64)
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_elem_user_info")]
extern "C" fn snd_ctl_elem_user_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
let ue: *mut user_element = snd_kcontrol_chip!(kcontrol) as *mut user_element;
let offset: u32 = snd_ctl_get_ioff(kcontrol, unsafe { core::ptr::addr_of_mut!((*uinfo).id) });
unsafe {
*uinfo = (*ue).info;
}
snd_ctl_build_ioff(unsafe { core::ptr::addr_of_mut!((*uinfo).id) }, kcontrol, offset);
0
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_elem_user_enum_info")]
extern "C" fn snd_ctl_elem_user_enum_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
let ue: *mut user_element = snd_kcontrol_chip!(kcontrol) as *mut user_element;
let mut names: *const u8 = unsafe { core::mem::zeroed() };
let mut item: u32 = unsafe { (*uinfo).value.enumerated.item };
let offset: u32 = snd_ctl_get_ioff(kcontrol, unsafe { core::ptr::addr_of_mut!((*uinfo).id) });
unsafe {
*uinfo = (*ue).info;
}
snd_ctl_build_ioff(unsafe { core::ptr::addr_of_mut!((*uinfo).id) }, kcontrol, offset);
let mut __v0: u32;
let __UNIQUE_ID_x__568: u32 = item;
let __UNIQUE_ID_y__569: u32 = unsafe { (*uinfo).value.enumerated.items }.wrapping_sub(1);
loop {
if __types_ok!(__UNIQUE_ID_x__568, __UNIQUE_ID_y__569) == 0 {
unsafe { __compiletime_assert_570() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__568 < __UNIQUE_ID_y__569 { __UNIQUE_ID_x__568 } else { __UNIQUE_ID_y__569 };
item = __v0;
unsafe {
(*uinfo).value.enumerated.item = item;
}
names = (unsafe { (*ue).priv_data }) as *const u8;
while item > 0 {
let __v2: u64 = unsafe { strlen(names) }.wrapping_add(1);
let __v3: *const u8 = names;
names = __v3.wrapping_offset(__v2 as isize);
item = item.wrapping_sub(1);
}
unsafe { sized_strscpy((*uinfo).value.enumerated.name.as_mut_ptr() as *mut u8, names, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_array__54!() as u64).wrapping_add(__must_be_cstr!() as u64).wrapping_add(__must_be_cstr__40!() as u64)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_elem_user_get")]
extern "C" fn snd_ctl_elem_user_get(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let ue: *mut user_element = snd_kcontrol_chip!(kcontrol) as *mut user_element;
let size: u32 = (unsafe { (*ue).elem_data_size }) as u32;
let src: *mut u8 = unsafe { (*ue).elem_data }.wrapping_offset(snd_ctl_get_ioff(kcontrol, unsafe { core::ptr::addr_of_mut!((*ucontrol).id) }).wrapping_mul(size) as isize);
unsafe { memcpy(core::ptr::addr_of_mut!((*ucontrol).value) as *mut (), src as *const (), size as u64) };
0
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_elem_user_put")]
extern "C" fn snd_ctl_elem_user_put(kcontrol: *mut snd_kcontrol, ucontrol: *mut snd_ctl_elem_value) -> i32 {
let mut err: i32 = 0;
let mut change: i32 = 0;
let ue: *mut user_element = snd_kcontrol_chip!(kcontrol) as *mut user_element;
let size: u32 = (unsafe { (*ue).elem_data_size }) as u32;
let dst: *mut u8 = unsafe { (*ue).elem_data }.wrapping_offset(snd_ctl_get_ioff(kcontrol, unsafe { core::ptr::addr_of_mut!((*ucontrol).id) }).wrapping_mul(size) as isize);
err = sanity_check_input_values(unsafe { (*ue).card }, ucontrol as *const snd_ctl_elem_value, (unsafe { core::ptr::addr_of_mut!((*ue).info) }) as *const snd_ctl_elem_info, false);
if err < 0 {
return err;
}
change = ((unsafe { memcmp(core::ptr::addr_of_mut!((*ucontrol).value) as *const (), dst as *const (), size as u64) }) != 0) as i32;
if change != 0 {
unsafe { memcpy(dst as *mut (), core::ptr::addr_of_mut!((*ucontrol).value) as *const (), size as u64) };
}
change
}

/// called in controls_rwsem write lock
fn replace_user_tlv(kctl: *mut snd_kcontrol, buf: *mut u32, size: u32) -> i32 {
let ue: *mut user_element = snd_kcontrol_chip!(kctl) as *mut user_element;
let mut mask: u32 = 0;
let mut i: i32 = 0;
let mut change: i32 = 0;
lockdep_assert_held_write!(unsafe { core::ptr::addr_of_mut!((*(*ue).card).controls_rwsem) });
if size > 131072u32 {
return 22i32.wrapping_neg();
}
// sane value
// does the TLV size change cause overflow?
if check_user_elem_overflow(unsafe { (*ue).card }, (size as u64).wrapping_sub(unsafe { (*ue).tlv_data_size }) as i64) {
return 12i32.wrapping_neg();
}
let mut container: *mut u32 = (unsafe { vmemdup_user(buf as *const (), size as u64) }) as *mut u32;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(container) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_kvfree(__slate_cleanup_ptr_1 as *mut _));
if IS_ERR(container as *const ()) {
return PTR_ERR(container as *const ()) as i32;
}
change = ((unsafe { (*ue).tlv_data_size }) != (size as u64)) as i32;
if change == 0 {
change = ((unsafe { memcmp((*ue).tlv_data as *const (), container as *const (), size as u64) }) != 0) as i32;
}
if change == 0 {
return 0;
}
if unsafe { (*ue).tlv_data }.is_null() {
// Now TLV data is available.
i = 0;
while (i as u32) < unsafe { (*kctl).count } {
let __v2: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(i as isize);
unsafe {
(*__v2).access = (*__v2).access | 16u32;
}
i = i.wrapping_add(1);
}
mask = 2u32;
} else {
let __v8: *mut snd_card = unsafe { (*ue).card };
let __v9: u64 = unsafe { (*__v8).user_ctl_alloc_size };
unsafe {
(*__v8).user_ctl_alloc_size = __v9.wrapping_sub((*ue).tlv_data_size);
}
unsafe {
(*ue).tlv_data_size = 0;
}
unsafe { kvfree((*ue).tlv_data as *const ()) };
}
unsafe {
(*ue).tlv_data = no_free_ptr__2!(container) as *mut ();
}
unsafe {
(*ue).tlv_data_size = size as u64;
}
// decremented at private_free.
let __v14: *mut snd_card = unsafe { (*ue).card };
let __v15: u64 = unsafe { (*__v14).user_ctl_alloc_size };
unsafe {
(*__v14).user_ctl_alloc_size = __v15.wrapping_add(size as u64);
}
mask |= 8u32;
i = 0;
while (i as u32) < unsafe { (*kctl).count } {
snd_ctl_notify_one(unsafe { (*ue).card }, mask, kctl, i as u32);
i = i.wrapping_add(1);
}
change
}

fn read_user_tlv(kctl: *mut snd_kcontrol, buf: *mut u32, size: u32) -> i32 {
let ue: *mut user_element = snd_kcontrol_chip!(kctl) as *mut user_element;
if (unsafe { (*ue).tlv_data_size }) == 0 || unsafe { (*ue).tlv_data }.is_null() {
return 6i32.wrapping_neg();
}
if (size as u64) < unsafe { (*ue).tlv_data_size } {
return 28i32.wrapping_neg();
}
if copy_to_user(buf as *mut (), (unsafe { (*ue).tlv_data }) as *const (), unsafe { (*ue).tlv_data_size }) != 0 {
return 14i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_elem_user_tlv")]
extern "C" fn snd_ctl_elem_user_tlv(kctl: *mut snd_kcontrol, op_flag: i32, size: u32, buf: *mut u32) -> i32 {
if op_flag == 1 {
return replace_user_tlv(kctl, buf, size);
} else {
return read_user_tlv(kctl, buf, size);
}
}

/// called in controls_rwsem write lock
fn snd_ctl_elem_init_enum_names(ue: *mut user_element) -> i32 {
let mut buf_len: u64 = 0;
let mut name_len: u64 = 0;
let mut i: u32 = 0;
let user_ptrval: u64 = unsafe { (*ue).info.value.enumerated.names_ptr };
lockdep_assert_held_write!(unsafe { core::ptr::addr_of_mut!((*(*ue).card).controls_rwsem) });
buf_len = (unsafe { (*ue).info.value.enumerated.names_length }) as u64;
if buf_len > 65536u64 {
return 22i32.wrapping_neg();
}
if check_user_elem_overflow(unsafe { (*ue).card }, buf_len as i64) {
return 12i32.wrapping_neg();
}
let mut names: *mut u8 = (unsafe { vmemdup_user(user_ptrval as *const (), buf_len) }) as *mut u8;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(names) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_kvfree(__slate_cleanup_ptr_1 as *mut _));
if IS_ERR(names as *const ()) {
return PTR_ERR(names as *const ()) as i32;
}
// check that there are enough valid names
let mut p: *mut u8 = names;
while i < unsafe { (*ue).info.value.enumerated.items } {
if buf_len == 0 {
return 22i32.wrapping_neg();
}
name_len = unsafe { strnlen(p as *const u8, buf_len) };
if name_len == 0 || name_len >= 64 || name_len == buf_len {
return 22i32.wrapping_neg();
}
let __v2: *mut u8 = p;
p = __v2.wrapping_offset(name_len.wrapping_add(1) as isize);
buf_len = buf_len.wrapping_sub(name_len.wrapping_add(1));
i = i.wrapping_add(1);
}
unsafe {
(*ue).priv_data = no_free_ptr__3!(names) as *mut ();
}
unsafe {
(*ue).info.value.enumerated.names_ptr = 0;
}
// increment the allocation size; decremented again at private_free.
let __v12: *mut snd_card = unsafe { (*ue).card };
let __v13: u64 = unsafe { (*__v12).user_ctl_alloc_size };
unsafe {
(*__v12).user_ctl_alloc_size = __v13.wrapping_add((*ue).info.value.enumerated.names_length as u64);
}
0
}

fn compute_user_elem_size(size: u64, count: u32) -> u64 {
(core::mem::size_of::<user_element>() as u64).wrapping_add(size.wrapping_mul(count as u64))
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_elem_user_free")]
extern "C" fn snd_ctl_elem_user_free(kcontrol: *mut snd_kcontrol) {
let ue: *mut user_element = snd_kcontrol_chip!(kcontrol) as *mut user_element;
// decrement the allocation size.
let __v0: u64 = compute_user_elem_size(unsafe { (*ue).elem_data_size }, unsafe { (*kcontrol).count });
let __v1: *mut snd_card = unsafe { (*ue).card };
let __v2: u64 = unsafe { (*__v1).user_ctl_alloc_size };
unsafe {
(*__v1).user_ctl_alloc_size = __v2.wrapping_sub(__v0);
}
let __v4: *mut snd_card = unsafe { (*ue).card };
let __v5: u64 = unsafe { (*__v4).user_ctl_alloc_size };
unsafe {
(*__v4).user_ctl_alloc_size = __v5.wrapping_sub((*ue).tlv_data_size);
}
if !unsafe { (*ue).priv_data }.is_null() {
let __v7: *mut snd_card = unsafe { (*ue).card };
let __v8: u64 = unsafe { (*__v7).user_ctl_alloc_size };
unsafe {
(*__v7).user_ctl_alloc_size = __v8.wrapping_sub((*ue).info.value.enumerated.names_length as u64);
}
}
unsafe { kvfree((*ue).tlv_data as *const ()) };
unsafe { kvfree((*ue).priv_data as *const ()) };
unsafe { kfree(ue as *const ()) };
}

fn snd_ctl_elem_add(file: *mut snd_ctl_file, info: *mut snd_ctl_elem_info, replace: i32) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let card: *mut snd_card = unsafe { (*file).card };
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut count: u32 = 0;
let mut access: u32 = 0;
let mut private_size: i64 = 0;
let mut alloc_size: u64 = 0;
let mut offset: u32 = 0;
let mut err: i32 = 0;
if (unsafe { *((*info).id.name.as_mut_ptr() as *mut u8) }) == 0 {
return 22i32.wrapping_neg();
}
if (unsafe { strnlen(((*info).id.name.as_mut_ptr() as *mut u8) as *const u8, core::mem::size_of::<[u8; 44]>() as u64) }) >= (core::mem::size_of::<[u8; 44]>() as u64) {
return 22i32.wrapping_neg();
}
// Delete a control to replace them if needed.
if replace != 0 {
unsafe {
(*info).id.numid = 0;
}
err = snd_ctl_remove_user_ctl(file, unsafe { core::ptr::addr_of_mut!((*info).id) });
if err != 0 {
return err;
}
}
// Check the number of elements for this userspace control.
count = (unsafe { (*info).owner }) as u32;
if count == 0 {
count = 1;
}
if count > 1028 {
return 22i32.wrapping_neg();
}
// Arrange access permissions if needed.
access = unsafe { (*info).access };
if access == 0 {
access = 3u32;
}
access &= 291u32;
// In initial state, nothing is available as TLV container.
if access & 32u32 != 0 {
access |= 268435456u32;
}
access |= 536870912u32;
// Check information and calculate the size of data specific to
// this userspace control.
//
// pass NULL to card for suppressing error messages
err = snd_ctl_check_elem_info(core::ptr::null_mut::<snd_card>(), info as *const snd_ctl_elem_info);
if err < 0 {
return err;
}
// user-space control doesn't allow zero-size data
if (unsafe { (*info).count }) < 1 {
return 22i32.wrapping_neg();
}
private_size = unsafe { value_sizes.0[(*info).r#type as usize] }.wrapping_mul(unsafe { (*info).count }) as i64;
alloc_size = compute_user_elem_size(private_size as u64, count);
let mut __UNIQUE_ID_guard_571: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_571) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_6 as *mut _));
let mut __UNIQUE_ID_unlock_572: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_572) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_7 as *mut _));
if check_user_elem_overflow(card, alloc_size as i64) {
return 12i32.wrapping_neg();
}
// Keep memory object for this userspace control. After passing this
// code block, the instance should be freed by snd_ctl_free_one().
//
// Note that these elements in this control are locked.
err = snd_ctl_new(unsafe { &mut (*core::ptr::addr_of_mut!(kctl)) }, count, access, file);
if err < 0 {
return err;
}
unsafe { memcpy(core::ptr::addr_of_mut!((*kctl).id) as *mut (), core::ptr::addr_of_mut!((*info).id) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) };
let mut __v8: *mut ();
let mut __v9: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(alloc_size, 3264u32, unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(alloc_size, 3264u32, unsafe { *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
} });
}
__v9 = _res;
__v8 = __v9;
let ue: *mut user_element = __v8 as *mut user_element;
if ue.is_null() {
unsafe { kfree(kctl as *const ()) };
return 12i32.wrapping_neg();
}
unsafe {
(*kctl).private_data = ue as *mut ();
}
unsafe {
(*kctl).private_free = Some(snd_ctl_elem_user_free);
}
// increment the allocated size; decremented again at private_free.
let __v14: u64 = unsafe { (*card).user_ctl_alloc_size };
unsafe {
(*card).user_ctl_alloc_size = __v14.wrapping_add(alloc_size);
}
// Set private data for this userspace control.
unsafe {
(*ue).card = card;
}
unsafe {
(*ue).info = *info;
}
unsafe {
(*ue).info.access = 0;
}
unsafe {
(*ue).elem_data = (ue as *mut u8).wrapping_offset(core::mem::size_of::<user_element>() as isize);
}
unsafe {
(*ue).elem_data_size = private_size as u64;
}
if (unsafe { (*ue).info.r#type }) == 3 {
err = snd_ctl_elem_init_enum_names(ue);
if err < 0 {
snd_ctl_free_one(kctl);
return err;
}
}
// Set callback functions.
if (unsafe { (*info).r#type }) == 3 {
unsafe {
(*kctl).info = Some(snd_ctl_elem_user_enum_info);
}
} else {
unsafe {
(*kctl).info = Some(snd_ctl_elem_user_info);
}
}
if access & 1u32 != 0 {
unsafe {
(*kctl).get = Some(snd_ctl_elem_user_get);
}
}
if access & 2u32 != 0 {
unsafe {
(*kctl).put = Some(snd_ctl_elem_user_put);
}
}
if access & 32u32 != 0 {
unsafe {
(*kctl).tlv.c = Some(snd_ctl_elem_user_tlv);
}
}
// This function manage to free the instance on failure.
err = __snd_ctl_add_replace(card, kctl, 0u32);
if err < 0 {
snd_ctl_free_one(kctl);
return err;
}
offset = snd_ctl_get_ioff(kctl, unsafe { core::ptr::addr_of_mut!((*info).id) });
snd_ctl_build_ioff(unsafe { core::ptr::addr_of_mut!((*info).id) }, kctl, offset);
// Here we cannot fill any field for the number of elements added by
// this operation because there're no specific fields. The usage of
// 'owner' field for this purpose may cause any bugs to userspace
// applications because the field originally means PID of a process
// which locks the element.
0
}

fn snd_ctl_elem_add_user(file: *mut snd_ctl_file, _info: *mut snd_ctl_elem_info, replace: i32) -> i32 {
let mut info: snd_ctl_elem_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 272usize);
}
let mut err: i32 = 0;
if copy_from_user(core::ptr::addr_of_mut!(info) as *mut (), _info as *const (), core::mem::size_of::<snd_ctl_elem_info>() as u64) != 0 {
return 14i32.wrapping_neg();
}
err = snd_ctl_elem_add(file, core::ptr::addr_of_mut!(info), replace);
if err < 0 {
return err;
}
if copy_to_user(_info as *mut (), core::ptr::addr_of_mut!(info) as *const (), core::mem::size_of::<snd_ctl_elem_info>() as u64) != 0 {
snd_ctl_remove_user_ctl(file, core::ptr::addr_of_mut!(info.id));
return 14i32.wrapping_neg();
}
0
}

fn snd_ctl_elem_remove(file: *mut snd_ctl_file, _id: *mut snd_ctl_elem_id) -> i32 {
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
if copy_from_user(core::ptr::addr_of_mut!(id) as *mut (), _id as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
snd_ctl_remove_user_ctl(file, core::ptr::addr_of_mut!(id))
}

fn snd_ctl_subscribe_events(file: *mut snd_ctl_file, mut ptr: *mut i32) -> i32 {
let mut subscribe: i32 = 0;
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") ptr as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
subscribe = __val_gu as i32;
__v1 = __ret_gu as i64;
__v0 = __v1;
if __v0 != 0 {
return 14i32.wrapping_neg();
}
if subscribe < 0 {
subscribe = unsafe { (*file).subscribed };
let mut __v4: i64;
might_fault();
let mut __v5: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = subscribe;
let __ptr: *mut i32 = ptr;
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t2: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t2, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
instrument_put_user!(__x, __ptr);
__v5 = __ret_pu as i64;
__v4 = __v5;
if __v4 != 0 {
return 14i32.wrapping_neg();
}
return 0;
}
if subscribe != 0 {
unsafe {
(*file).subscribed = 1;
}
return 0;
} else {
if (unsafe { (*file).subscribed }) != 0 {
snd_ctl_empty_read_queue(file);
unsafe {
(*file).subscribed = 0;
}
}
}
0
}

fn call_tlv_handler(file: *mut snd_ctl_file, op_flag: i32, kctl: *mut snd_kcontrol, id: &snd_ctl_elem_id, buf: *mut u32, size: u32) -> i32 {
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(snd_ctl_get_ioff(kctl, (id as *const snd_ctl_elem_id) as *mut snd_ctl_elem_id) as isize);
// Check support of the request for this element.
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__35!() {
if op_flag == unsafe { pairs.0[i as usize].op } && (unsafe { (*vd).access }) & ((unsafe { pairs.0[i as usize].perm }) as u32) != 0 {
break;
}
i = i.wrapping_add(1);
}
if (i as u64) == ARRAY_SIZE__37!() {
return 6i32.wrapping_neg();
}
if (unsafe { (*kctl).tlv.c }) == None {
return 6i32.wrapping_neg();
}
// Write and command operations are not allowed for locked element.
if op_flag != 0 && !unsafe { (*vd).owner }.is_null() && (unsafe { (*vd).owner }) != file {
return 1i32.wrapping_neg();
}
unsafe { unsafe { (*kctl).tlv.c }.unwrap()(kctl, op_flag, size, buf) }
}

static pairs: __SlateAlign16<[__SlateRecord_control_c_1931; 3]> = __SlateAlign16([__SlateRecord_control_c_1931 { op: 0i32, perm: 16i32 }, __SlateRecord_control_c_1931 { op: 1i32, perm: 32i32 }, __SlateRecord_control_c_1931 { op: -1i32, perm: 64i32 }]);

fn read_tlv_buf(kctl: *mut snd_kcontrol, id: &snd_ctl_elem_id, buf: *mut u32, size: u32) -> i32 {
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(snd_ctl_get_ioff(kctl, (id as *const snd_ctl_elem_id) as *mut snd_ctl_elem_id) as isize);
let mut len: u32 = 0;
if (unsafe { (*vd).access }) & 16u32 == 0 {
return 6i32.wrapping_neg();
}
if unsafe { (*kctl).tlv.p }.is_null() {
return 6i32.wrapping_neg();
}
len = (core::mem::size_of::<u32>() as u64).wrapping_mul(2).wrapping_add((unsafe { *unsafe { (*kctl).tlv.p }.wrapping_offset(1 as isize) }) as u64) as u32;
if size < len {
return 12i32.wrapping_neg();
}
if copy_to_user(buf as *mut (), (unsafe { (*kctl).tlv.p }) as *const (), len as u64) != 0 {
return 14i32.wrapping_neg();
}
0
}

fn snd_ctl_tlv_ioctl(file: *mut snd_ctl_file, buf: *mut snd_ctl_tlv, op_flag: i32) -> i32 {
let mut header: snd_ctl_tlv = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(header) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut container_size: u32 = 0;
let mut kctl: *mut snd_kcontrol = unsafe { core::mem::zeroed() };
let mut id: snd_ctl_elem_id = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(id) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*(*file).card).controls_rwsem) });
if copy_from_user(core::ptr::addr_of_mut!(header) as *mut (), buf as *const (), 8u64) != 0 {
return 14i32.wrapping_neg();
}
// In design of control core, numerical ID starts at 1.
if header.numid == 0 {
return 22i32.wrapping_neg();
}
// At least, container should include type and length fields.
if (header.length as u64) < (core::mem::size_of::<u32>() as u64).wrapping_mul(2) {
return 22i32.wrapping_neg();
}
container_size = header.length;
let container: *mut u32 = unsafe { core::ptr::addr_of_mut!((*buf).tlv) as *mut u32 };
kctl = snd_ctl_find_numid(unsafe { (*file).card }, header.numid);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
// Calculate index of the element in this set.
id = unsafe { (*kctl).id };
snd_ctl_build_ioff(core::ptr::addr_of_mut!(id), kctl, header.numid.wrapping_sub(id.numid));
let vd: *mut snd_kcontrol_volatile = unsafe { core::ptr::addr_of_mut!((*kctl).vd) as *mut snd_kcontrol_volatile }.wrapping_offset(snd_ctl_get_ioff(kctl, core::ptr::addr_of_mut!(id)) as isize);
if (unsafe { (*vd).access }) & 268435456u32 != 0 {
return call_tlv_handler(file, op_flag, kctl, unsafe { &(*core::ptr::addr_of_mut!(id)) }, container, container_size);
} else {
if op_flag == 0 {
return read_tlv_buf(kctl, unsafe { &(*core::ptr::addr_of_mut!(id)) }, container, container_size);
}
}
// Not supported.
6i32.wrapping_neg()
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_ioctl")]
extern "C" fn snd_ctl_ioctl(file: *mut file, cmd: u32, arg: u64) -> i64 {
let mut __UNIQUE_ID_unlock_580: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_580) as *mut *mut __SlateRecord_rwsem_h_259;
let mut __UNIQUE_ID_guard_579: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_579) as *mut __SlateRecord_rwsem_h_259;
let mut __x: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__val_pu) as *mut i32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __UNIQUE_ID_unlock_578: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_578) as *mut *mut __SlateRecord_rwsem_h_270;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_270;
let mut __UNIQUE_ID_unlock_576: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_576) as *mut *mut __SlateRecord_rwsem_h_270;
let mut scope_1: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(scope_1) as *mut __SlateRecord_rwsem_h_270;
let mut __UNIQUE_ID_unlock_574: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_574) as *mut *mut __SlateRecord_rwsem_h_259;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_259;
let mut __x_1: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(__val_pu) as *mut i32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut ip: *mut i32 = unsafe { core::mem::zeroed() };
let mut argp: *mut () = unsafe { core::mem::zeroed() };
let mut p: core::mem::MaybeUninit<*mut snd_kctl_ioctl> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut snd_kctl_ioctl = core::ptr::addr_of_mut!(p) as *mut *mut snd_kctl_ioctl;
let mut card: core::mem::MaybeUninit<*mut snd_card> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut snd_card = core::ptr::addr_of_mut!(card) as *mut *mut snd_card;
let mut ctl: core::mem::MaybeUninit<*mut snd_ctl_file> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut snd_ctl_file = core::ptr::addr_of_mut!(ctl) as *mut *mut snd_ctl_file;
let __slate_cleanup_ptr_14 = __slate_slot_14;
let __slate_cleanup_armed_14 = core::cell::Cell::new(false);
let __slate_cleanup_14 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_14.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_14 as *mut _) };
}
});
let __slate_cleanup_ptr_13 = __slate_slot_13;
let __slate_cleanup_armed_13 = core::cell::Cell::new(false);
let __slate_cleanup_13 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_13.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_13 as *mut _) };
}
});
let __slate_cleanup_ptr_12 = __slate_slot_12;
let __slate_cleanup_armed_12 = core::cell::Cell::new(false);
let __slate_cleanup_12 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_12.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_12 as *mut _) };
}
});
let __slate_cleanup_ptr_11 = __slate_slot_11;
let __slate_cleanup_armed_11 = core::cell::Cell::new(false);
let __slate_cleanup_11 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_11.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_11 as *mut _) };
}
});
let __slate_cleanup_ptr_10 = __slate_slot_10;
let __slate_cleanup_armed_10 = core::cell::Cell::new(false);
let __slate_cleanup_10 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_10.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_10 as *mut _) };
}
});
let __slate_cleanup_ptr_9 = __slate_slot_9;
let __slate_cleanup_armed_9 = core::cell::Cell::new(false);
let __slate_cleanup_9 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_9.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_9 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_4 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
argp = arg as *mut ();
ip = argp as *mut i32;
let __slate_auto_bytes_7 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
*__slate_slot_27 = (*file).private_data as *mut snd_ctl_file;
*__slate_slot_26 = (*(*__slate_slot_27)).card;
if snd_BUG_ON!((*__slate_slot_26).is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
} else {
let __t2: u32 = cmd;
if __t2 == 2147767552 {
might_fault();
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_1 = SNDRV_PROTOCOL_VERSION!(2i32, 0i32, 10i32);
let __ptr: *mut i32 = ip;
__chk_user_ptr!();
*__slate_slot_18 = __ptr as *mut ();
*__slate_slot_17 = __x_1;
let __t1: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_18 as u64 => __t1, in("ax") *__slate_slot_17 as u32, options(att_syntax));
*__slate_slot_19 = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
instrument_put_user!(__x_1, __ptr);
return (if (*__slate_slot_19 as i64) != 0 { 14i32.wrapping_neg() } else { 0i32 }) as i64;
} else {
if __t2 == 2172146945 {
return snd_ctl_card_info(*__slate_slot_26, *__slate_slot_27, cmd, argp) as i64;
} else {
if __t2 == 3222820098 {
return snd_ctl_card_bytes_user(*__slate_slot_26, argp as *mut snd_ctl_card_bytes) as i64;
} else {
if __t2 == 3226490128 {
return snd_ctl_elem_list_user(*__slate_slot_26, argp as *mut snd_ctl_elem_list) as i64;
} else {
if __t2 == 3239073041 {
return snd_ctl_elem_info_user(*__slate_slot_27, argp as *mut snd_ctl_elem_info) as i64;
} else {
if __t2 == 3301463314 {
return snd_ctl_elem_read_user(*__slate_slot_26, argp as *mut snd_ctl_elem_value) as i64;
} else {
if __t2 == 3301463315 {
return snd_ctl_elem_write_user(*__slate_slot_27, argp as *mut snd_ctl_elem_value) as i64;
} else {
if __t2 == 1077957908 {
return snd_ctl_elem_lock(*__slate_slot_27, argp as *mut snd_ctl_elem_id) as i64;
} else {
if __t2 == 1077957909 {
return snd_ctl_elem_unlock(*__slate_slot_27, argp as *mut snd_ctl_elem_id) as i64;
} else {
if __t2 == 3239073047 {
return snd_ctl_elem_add_user(*__slate_slot_27, argp as *mut snd_ctl_elem_info, 0i32) as i64;
} else {
if __t2 == 3239073048 {
return snd_ctl_elem_add_user(*__slate_slot_27, argp as *mut snd_ctl_elem_info, 1i32) as i64;
} else {
if __t2 == 3225441561 {
return snd_ctl_elem_remove(*__slate_slot_27, argp as *mut snd_ctl_elem_id) as i64;
} else {
if __t2 == 3221509398 {
return snd_ctl_subscribe_events(*__slate_slot_27, ip) as i64;
} else {
if __t2 == 3221771546 {
*__slate_slot_22 = snd_power_ref_and_wait(*__slate_slot_26);
if *__slate_slot_22 < 0 {
return *__slate_slot_22 as i64;
} else {
core::ptr::write(__slate_slot_14, class_rwsem_read_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_26)).controls_rwsem)));
__slate_cleanup_armed_14.set(true);
core::ptr::write(__slate_slot_13, (core::ptr::addr_of_mut!((*(*__slate_slot_26)).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_13.set(true);
if !class_rwsem_read_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_14))).is_null() || !class_rwsem_read_is_conditional {
if true {
*__slate_slot_22 = snd_ctl_tlv_ioctl(*__slate_slot_27, argp as *mut snd_ctl_tlv, 0i32);
}
}
__slate_cleanup_armed_13.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_13 as *mut _);
__slate_cleanup_armed_14.set(false);
class_rwsem_read_destructor(__slate_slot_14 as *mut _);
snd_power_unref(*__slate_slot_26);
return *__slate_slot_22 as i64;
}
} else {
if __t2 == 3221771547 {
*__slate_slot_22 = snd_power_ref_and_wait(*__slate_slot_26);
if *__slate_slot_22 < 0 {
return *__slate_slot_22 as i64;
} else {
core::ptr::write(__slate_slot_12, class_rwsem_write_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_26)).controls_rwsem)));
__slate_cleanup_armed_12.set(true);
core::ptr::write(__slate_slot_11, (core::ptr::addr_of_mut!((*(*__slate_slot_26)).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_11.set(true);
if !class_rwsem_write_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_12))).is_null() || !class_rwsem_write_is_conditional {
if true {
*__slate_slot_22 = snd_ctl_tlv_ioctl(*__slate_slot_27, argp as *mut snd_ctl_tlv, 1i32);
}
}
__slate_cleanup_armed_11.set(false);
__class_rwsem_write_cleanup_ctx(__slate_slot_11 as *mut _);
__slate_cleanup_armed_12.set(false);
class_rwsem_write_destructor(__slate_slot_12 as *mut _);
snd_power_unref(*__slate_slot_26);
return *__slate_slot_22 as i64;
}
} else {
if __t2 == 3221771548 {
*__slate_slot_22 = snd_power_ref_and_wait(*__slate_slot_26);
if *__slate_slot_22 < 0 {
return *__slate_slot_22 as i64;
} else {
core::ptr::write(__slate_slot_10, class_rwsem_write_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_26)).controls_rwsem)));
__slate_cleanup_armed_10.set(true);
core::ptr::write(__slate_slot_9, (core::ptr::addr_of_mut!((*(*__slate_slot_26)).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_9.set(true);
if !class_rwsem_write_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_10))).is_null() || !class_rwsem_write_is_conditional {
if true {
*__slate_slot_22 = snd_ctl_tlv_ioctl(*__slate_slot_27, argp as *mut snd_ctl_tlv, -1i32);
}
}
__slate_cleanup_armed_9.set(false);
__class_rwsem_write_cleanup_ctx(__slate_slot_9 as *mut _);
__slate_cleanup_armed_10.set(false);
class_rwsem_write_destructor(__slate_slot_10 as *mut _);
snd_power_unref(*__slate_slot_26);
return *__slate_slot_22 as i64;
}
} else {
if __t2 == 3221509584 {
return 92i32.wrapping_neg() as i64;
} else {
if __t2 == 2147767761 {
might_fault();
let __slate_auto_bytes_8 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x = 0;
let __ptr: *mut i32 = ip;
__chk_user_ptr!();
*__slate_slot_5 = __ptr as *mut ();
*__slate_slot_4 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_5 as u64 => __t0, in("ax") *__slate_slot_4 as u32, options(att_syntax));
*__slate_slot_6 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user!(__x, __ptr);
return (if (*__slate_slot_6 as i64) != 0 { 14i32.wrapping_neg() } else { 0i32 }) as i64;
} else {
core::ptr::write(__slate_slot_1, class_rwsem_read_constructor(core::ptr::addr_of_mut!(snd_ioctl_rwsem)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(snd_ioctl_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_0.set(true);
*__slate_slot_25 = list_first_entry__12!(core::ptr::addr_of_mut!(snd_control_ioctls));
while list_entry_is_head!(*__slate_slot_25, core::ptr::addr_of_mut!(snd_control_ioctls), list) == 0 {
*__slate_slot_22 = unsafe { (*(*__slate_slot_25)).fioctl }.unwrap()(*__slate_slot_26, *__slate_slot_27, cmd, arg);
if *__slate_slot_22 != 515i32.wrapping_neg() {
return *__slate_slot_22 as i64;
} else {
*__slate_slot_25 = list_next_entry__13!(*__slate_slot_25, list);
}
}
return 25i32.wrapping_neg() as i64;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_read")]
extern "C" fn snd_ctl_read(file: *mut file, mut buffer: *mut u8, mut count: u64, offset: *mut i64) -> i64 {
let mut __ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(__ret) as *mut u32;
let mut __ai_ptr: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut u32 = core::ptr::addr_of_mut!(__ai_ptr) as *mut *mut u32;
let mut wait: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut wait_queue_entry = core::ptr::addr_of_mut!(wait) as *mut wait_queue_entry;
let mut kev: core::mem::MaybeUninit<*mut snd_kctl_event> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut snd_kctl_event = core::ptr::addr_of_mut!(kev) as *mut *mut snd_kctl_event;
let mut ev: core::mem::MaybeUninit<snd_ctl_event> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut snd_ctl_event = core::ptr::addr_of_mut!(ev) as *mut snd_ctl_event;
let mut result: i64 = unsafe { core::mem::zeroed() };
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut ctl: core::mem::MaybeUninit<*mut snd_ctl_file> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut snd_ctl_file = core::ptr::addr_of_mut!(ctl) as *mut *mut snd_ctl_file;
unsafe {
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
err = 0;
result = 0;
*__slate_slot_15 = (*file).private_data as *mut snd_ctl_file;
if snd_BUG_ON!((*__slate_slot_15).is_null() || unsafe { (*(*__slate_slot_15)).card }.is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
} else {
if (*(*__slate_slot_15)).subscribed == 0 {
return 77i32.wrapping_neg() as i64;
} else {
if count < (core::mem::size_of::<snd_ctl_event>() as u64) {
return 22i32.wrapping_neg() as i64;
} else {
spin_lock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_15)).read_lock));
'__join_0: {
'__join_1: {
'__join_18: {
loop {
if count >= (core::mem::size_of::<snd_ctl_event>() as u64) {
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
while list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_15)).events) as *const list_head) != 0 {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
if (*file).f_flags & 2048u32 != 0 || result > 0 {
break '__join_18;
} else {
init_waitqueue_entry(&mut (*core::ptr::addr_of_mut!(*__slate_slot_10)), get_current());
add_wait_queue(core::ptr::addr_of_mut!((*(*__slate_slot_15)).change_sleep), core::ptr::addr_of_mut!(*__slate_slot_10));
loop {
debug_normal_state_change!();
trace_set_current_state!(1i32);
loop {
kcsan_mb!();
loop {
core::ptr::write(__slate_slot_8, core::ptr::addr_of_mut!((*get_current()).__state));
kcsan_mb!();
instrument_atomic_read_write(*__slate_slot_8 as *const (), core::mem::size_of::<u32>() as u64);
core::ptr::write(__slate_slot_6, 1u32);
let __t0: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) *__slate_slot_6 as u32 => __t0, in(reg) *__slate_slot_8, options(nostack, att_syntax));
*__slate_slot_6 = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
spin_unlock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_15)).read_lock));
schedule();
remove_wait_queue(core::ptr::addr_of_mut!((*(*__slate_slot_15)).change_sleep), core::ptr::addr_of_mut!(*__slate_slot_10));
if (*(*(*__slate_slot_15)).card).shutdown != 0 {
return 19i32.wrapping_neg() as i64;
} else {
if signal_pending(get_current()) != 0 {
return 512i32.wrapping_neg() as i64;
} else {
spin_lock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_15)).read_lock));
}
}
}
}
*__slate_slot_11 = snd_kctl_event!(unsafe { (*(*__slate_slot_15)).events.next });
(*__slate_slot_12).r#type = 0;
(*__slate_slot_12).data.elem.mask = (*(*__slate_slot_11)).mask;
(*__slate_slot_12).data.elem.id = (*(*__slate_slot_11)).id;
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_11)).list));
spin_unlock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_15)).read_lock));
kfree(*__slate_slot_11 as *const ());
if copy_to_user(buffer as *mut (), core::ptr::addr_of_mut!(*__slate_slot_12) as *const (), core::mem::size_of::<snd_ctl_event>() as u64) != 0 {
break;
} else {
spin_lock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_15)).read_lock));
let __v5: *mut u8 = buffer;
buffer = __v5.wrapping_offset(core::mem::size_of::<snd_ctl_event>() as isize);
count = count.wrapping_sub(core::mem::size_of::<snd_ctl_event>() as u64);
result = (result as u64).wrapping_add(core::mem::size_of::<snd_ctl_event>() as u64) as i64;
}
} else {
break '__join_1;
}
}
err = 14i32.wrapping_neg();
break '__join_0;
}
err = 11i32.wrapping_neg();
}
spin_unlock_irq(core::ptr::addr_of_mut!((*(*__slate_slot_15)).read_lock));
}
return if result > 0 { result } else { err as i64 };
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_poll")]
extern "C" fn snd_ctl_poll(file: *mut file, wait: *mut poll_table_struct) -> u32 {
let mut mask: u32 = 0;
let mut ctl: *mut snd_ctl_file = unsafe { core::mem::zeroed() };
ctl = (unsafe { (*file).private_data }) as *mut snd_ctl_file;
if (unsafe { (*ctl).subscribed }) == 0 {
return 0;
}
poll_wait(file, unsafe { core::ptr::addr_of_mut!((*ctl).change_sleep) }, wait);
if list_empty((unsafe { core::ptr::addr_of_mut!((*ctl).events) }) as *const list_head) == 0 {
mask |= 1 | 64;
}
mask
}

/// register the device-specific control-ioctls.
/// called from each device manager like pcm.c, hwdep.c, etc.
fn _snd_ctl_register_ioctl(fcn: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>, lists: *mut list_head) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut pn: *mut snd_kctl_ioctl = unsafe { core::mem::zeroed() };
let mut __v0: *mut snd_kctl_ioctl;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_kctl_ioctl>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut snd_kctl_ioctl;
pn = __v0;
if pn.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*pn).fioctl = fcn;
}
let mut __UNIQUE_ID_guard_581: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!(snd_ioctl_rwsem) });
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_581) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_6 as *mut _));
let mut __UNIQUE_ID_unlock_582: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!(snd_ioctl_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_582) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_7 as *mut _));
list_add_tail(unsafe { core::ptr::addr_of_mut!((*pn).list) }, lists);
0
}

/// snd_ctl_register_ioctl - register the device-specific control-ioctls
/// @fcn: ioctl callback function
///
/// called from each device manager like pcm.c, hwdep.c, etc.
///
/// Return: zero if successful, or a negative error code
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_register_ioctl")]
extern "C" fn snd_ctl_register_ioctl(fcn: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32 {
_snd_ctl_register_ioctl(fcn, unsafe { core::ptr::addr_of_mut!(snd_control_ioctls) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_register_ioctl_583: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32>, *mut ()>(Some(snd_ctl_register_ioctl)) };

/// snd_ctl_register_ioctl_compat - register the device-specific 32bit compat
/// control-ioctls
/// @fcn: ioctl callback function
///
/// Return: zero if successful, or a negative error code
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_register_ioctl_compat")]
extern "C" fn snd_ctl_register_ioctl_compat(fcn: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32 {
_snd_ctl_register_ioctl(fcn, unsafe { core::ptr::addr_of_mut!(snd_control_compat_ioctls) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_register_ioctl_compat_584: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32>, *mut ()>(Some(snd_ctl_register_ioctl_compat)) };

/// de-register the device-specific control-ioctls.
fn _snd_ctl_unregister_ioctl(fcn: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>, lists: *mut list_head) -> i32 {
let mut p: *mut snd_kctl_ioctl = unsafe { core::mem::zeroed() };
if snd_BUG_ON!(fcn == None) != 0 {
return 22i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_585: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!(snd_ioctl_rwsem) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_585) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_586: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!(snd_ioctl_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_586) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
p = list_first_entry__12!(lists);
while list_entry_is_head!(p, lists, list) == 0 {
if (unsafe { (*p).fioctl }) == fcn {
list_del(unsafe { core::ptr::addr_of_mut!((*p).list) });
unsafe { kfree(p as *const ()) };
return 0;
}
p = list_next_entry__13!(p, list);
}
snd_BUG!();
22i32.wrapping_neg()
}

/// snd_ctl_unregister_ioctl - de-register the device-specific control-ioctls
/// @fcn: ioctl callback function to unregister
///
/// Return: zero if successful, or a negative error code
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_unregister_ioctl")]
extern "C" fn snd_ctl_unregister_ioctl(fcn: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32 {
_snd_ctl_unregister_ioctl(fcn, unsafe { core::ptr::addr_of_mut!(snd_control_ioctls) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_unregister_ioctl_587: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32>, *mut ()>(Some(snd_ctl_unregister_ioctl)) };

/// snd_ctl_unregister_ioctl_compat - de-register the device-specific compat
/// 32bit control-ioctls
/// @fcn: ioctl callback function to unregister
///
/// Return: zero if successful, or a negative error code
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_unregister_ioctl_compat")]
extern "C" fn snd_ctl_unregister_ioctl_compat(fcn: Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32 {
_snd_ctl_unregister_ioctl(fcn, unsafe { core::ptr::addr_of_mut!(snd_control_compat_ioctls) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_unregister_ioctl_compat_588: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut snd_card, *mut snd_ctl_file, u32, u64) -> i32>) -> i32>, *mut ()>(Some(snd_ctl_unregister_ioctl_compat)) };

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_fasync")]
extern "C" fn snd_ctl_fasync(fd: i32, file: *mut file, on: i32) -> i32 {
let mut ctl: *mut snd_ctl_file = unsafe { core::mem::zeroed() };
ctl = (unsafe { (*file).private_data }) as *mut snd_ctl_file;
unsafe { snd_fasync_helper(fd, file, on, core::ptr::addr_of_mut!((*ctl).fasync)) }
}

/// return the preferred subdevice number if already assigned;
/// otherwise return -1
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_get_preferred_subdevice")]
extern "C" fn snd_ctl_get_preferred_subdevice(card: *mut snd_card, r#type: i32) -> i32 {
let mut kctl: *mut snd_ctl_file = unsafe { core::mem::zeroed() };
let mut subdevice: i32 = 1i32.wrapping_neg();
let mut __UNIQUE_ID_guard_589: __SlateRecord_spinlock_h_670 = class_read_lock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwlock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_589) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_read_lock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_590: *mut __SlateRecord_spinlock_h_670 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwlock) }) as u64) as *mut __SlateRecord_spinlock_h_670;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_590) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
kctl = list_first_entry__11!(unsafe { core::ptr::addr_of_mut!((*card).ctl_files) });
while list_entry_is_head!(kctl, unsafe { core::ptr::addr_of_mut!((*card).ctl_files) }, list) == 0 {
if (unsafe { (*kctl).pid }) == task_pid(get_current()) {
subdevice = unsafe { *((*kctl).preferred_subdevice.as_mut_ptr() as *mut i32).wrapping_offset(r#type as isize) };
if subdevice != 1i32.wrapping_neg() {
break;
}
}
kctl = list_next_entry__14!(kctl, list);
}
subdevice
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_get_preferred_subdevice_591: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, i32) -> i32>, *mut ()>(Some(snd_ctl_get_preferred_subdevice)) };

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_list32 {
    offset: u32,
    space: u32,
    used: u32,
    count: u32,
    pids: u32,
    reserved: [u8; 50],
}


fn snd_ctl_elem_list_compat(card: *mut snd_card, mut data32: *mut snd_ctl_elem_list32) -> i32 {
let mut data: snd_ctl_elem_list = {
let mut __t0: snd_ctl_elem_list = unsafe { core::mem::zeroed() };
__t0
};
let mut ptr: u32 = 0;
let mut err: i32 = 0;
// offset, space, used, count
if copy_from_user(core::ptr::addr_of_mut!(data) as *mut (), data32 as *const (), 4u64.wrapping_mul(core::mem::size_of::<u32>() as u64)) != 0 {
return 14i32.wrapping_neg();
}
// pids
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t1: u64;
let __t2: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).pids) as u64 => __t1, lateout("dx") __t2, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
instrument_get_user!(__val_gu);
ptr = __val_gu;
__v1 = __ret_gu as i64;
__v0 = __v1;
if __v0 != 0 {
return 14i32.wrapping_neg();
}
data.pids = compat_ptr(ptr) as *mut snd_ctl_elem_id;
err = snd_ctl_elem_list(card, unsafe { &mut (*core::ptr::addr_of_mut!(data)) });
if err < 0 {
return err;
}
// copy the result
if copy_to_user(data32 as *mut (), core::ptr::addr_of_mut!(data) as *const (), 4u64.wrapping_mul(core::mem::size_of::<u32>() as u64)) != 0 {
return 14i32.wrapping_neg();
}
0
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_info32 {
    id: snd_ctl_elem_id,
    r#type: i32,
    access: u32,
    count: u32,
    owner: i32,
    value: __SlateRecord_control_compat_c_56,
    reserved: [u8; 64],
}


fn snd_ctl_elem_info_compat(ctl: *mut snd_ctl_file, mut data32: *mut snd_ctl_elem_info32) -> i32 {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let card: *mut snd_card = unsafe { (*ctl).card };
let mut err: i32 = 0;
let mut data: *mut snd_ctl_elem_info = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(data) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut snd_ctl_elem_info;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_ctl_elem_info>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut snd_ctl_elem_info;
data = __v1;
if data.is_null() {
return 12i32.wrapping_neg();
}
// copy id
if copy_from_user((unsafe { core::ptr::addr_of_mut!((*data).id) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).id) }) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
// we need to copy the item index.
// hope this doesn't break anything..
let mut __v7: i64;
might_fault();
let mut __v8: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).value.enumerated.item) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
(*data).value.enumerated.item = __val_gu;
}
__v8 = __ret_gu as i64;
__v7 = __v8;
if __v7 != 0 {
return 14i32.wrapping_neg();
}
err = unsafe { snd_power_ref_and_wait(card) };
if err < 0 {
return err;
}
err = snd_ctl_elem_info(ctl, data);
snd_power_unref(card);
if err < 0 {
return err;
}
// restore info to 32bit
//
// id, type, access, count
let __v11: bool = copy_to_user((unsafe { core::ptr::addr_of_mut!((*data32).id) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data).id) }) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 || copy_to_user((unsafe { core::ptr::addr_of_mut!((*data32).r#type) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data).r#type) }) as *const (), 3u64.wrapping_mul(core::mem::size_of::<u32>() as u64)) != 0;
if __v11 {
return 14i32.wrapping_neg();
}
let mut __v12: i64;
might_fault();
let mut __v13: i64;
{
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = unsafe { (*data).owner };
let __ptr: *mut i32 = unsafe { core::ptr::addr_of_mut!((*data32).owner) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t2: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t2, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
instrument_put_user!(__x, __ptr);
__v13 = __ret_pu as i64;
}
__v12 = __v13;
if __v12 != 0 {
return 14i32.wrapping_neg();
}
match unsafe { (*data).r#type } {
1 | 2 => {
let mut __v14: i64;
might_fault();
let mut __v15: i64;
{
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = (unsafe { (*data).value.integer.min }) as i32;
let __ptr: *mut i32 = unsafe { core::ptr::addr_of_mut!((*data32).value.integer.min) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t3: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t3, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
instrument_put_user!(__x, __ptr);
__v15 = __ret_pu as i64;
}
__v14 = __v15;
let __v16: bool = if __v14 != 0 {
true
} else {
let mut __v17: i64;
might_fault();
let mut __v18: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = (unsafe { (*data).value.integer.max }) as i32;
let __ptr: *mut i32 = unsafe { core::ptr::addr_of_mut!((*data32).value.integer.max) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t4: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t4, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t4 as u32) };
}
instrument_put_user!(__x, __ptr);
__v18 = __ret_pu as i64;
__v17 = __v18;
__v17 != 0
};
let __v19: bool = if __v16 {
true
} else {
let mut __v20: i64;
might_fault();
let mut __v21: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = (unsafe { (*data).value.integer.step }) as i32;
let __ptr: *mut i32 = unsafe { core::ptr::addr_of_mut!((*data32).value.integer.step) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t5: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t5, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t5 as u32) };
}
instrument_put_user!(__x, __ptr);
__v21 = __ret_pu as i64;
__v20 = __v21;
__v20 != 0
};
if __v19 {
return 14i32.wrapping_neg();
}
}
6 => {
if copy_to_user((unsafe { core::ptr::addr_of_mut!((*data32).value.integer64) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data).value.integer64) }) as *const (), core::mem::size_of::<__SlateRecord_asound_h_1170>() as u64) != 0 {
return 14i32.wrapping_neg();
}
}
3 => {
if copy_to_user((unsafe { core::ptr::addr_of_mut!((*data32).value.enumerated) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data).value.enumerated) }) as *const (), core::mem::size_of::<__SlateRecord_asound_h_1175>() as u64) != 0 {
return 14i32.wrapping_neg();
}
}
_ => {
}
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_ctl_elem_value32 {
    id: snd_ctl_elem_id,
    indirect: u32,
    value: __SlateRecord_control_compat_c_143,
    reserved: [u8; 128],
}


fn get_ctl_type(card: *mut snd_card, id: *mut snd_ctl_elem_id, countp: &mut i32) -> i32 {
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
let mut __UNIQUE_ID_guard_592: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_592) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_593: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_593) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let kctl: *mut snd_kcontrol = snd_ctl_find_id(card, id as *const snd_ctl_elem_id);
if kctl.is_null() {
return 2i32.wrapping_neg();
}
let mut info: *mut snd_ctl_elem_info = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(info) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_2 as *mut _));
let mut __v3: *mut snd_ctl_elem_info;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_ctl_elem_info>() as u64, 1u64);
let mut __v4: *mut ();
let mut __v5: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_8 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_8)
} });
}
__v5 = _res;
__v4 = __v5;
__v3 = __v4 as *mut snd_ctl_elem_info;
info = __v3;
if info.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*info).id = *id;
}
err = unsafe { unsafe { (*kctl).info }.unwrap()(kctl, info) };
if err >= 0 {
err = unsafe { (*info).r#type };
unsafe {
*(countp as *mut i32) = (*info).count as i32;
}
}
err
}

fn get_elem_size(r#type: i32, count: i32) -> i32 {
match r#type {
6 => {
return (core::mem::size_of::<i64>() as u64).wrapping_mul(count as u64) as i32;
}
3 => {
return (core::mem::size_of::<i32>() as u64).wrapping_mul(count as u64) as i32;
}
4 => {
return 512;
}
5 => {
return core::mem::size_of::<snd_aes_iec958>() as i32;
}
_ => {
return 1i32.wrapping_neg();
}
}
unsafe { core::mem::zeroed() }
}

fn copy_ctl_value_from_user(card: *mut snd_card, data: *mut snd_ctl_elem_value, userdata: *mut (), valuep: *mut (), typep: &mut i32, countp: &mut i32) -> i32 {
let mut data32: *mut snd_ctl_elem_value32 = userdata as *mut snd_ctl_elem_value32;
let mut i: i32 = 0;
let mut r#type: i32 = 0;
let mut size: i32 = 0;
let mut count: i32 = 0;
let mut indirect: u32 = 0;
if copy_from_user((unsafe { core::ptr::addr_of_mut!((*data).id) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).id) }) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
let mut __v0: i64;
might_fault();
let mut __v1: i64;
{
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).indirect) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
indirect = __val_gu;
__v1 = __ret_gu as i64;
}
__v0 = __v1;
if __v0 != 0 {
return 14i32.wrapping_neg();
}
if indirect != 0 {
return 22i32.wrapping_neg();
}
r#type = get_ctl_type(card, unsafe { core::ptr::addr_of_mut!((*data).id) }, unsafe { &mut (*core::ptr::addr_of_mut!(count)) });
if r#type < 0 {
return r#type;
}
if r#type == 1 || r#type == 2 {
i = 0;
while i < count {
let mut intp: *mut i32 = valuep as *mut i32;
let mut val: i32 = 0;
let mut __v4: i64;
might_fault();
let mut __v5: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t2: u64;
let __t3: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") intp.wrapping_offset(i as isize) as u64 => __t2, lateout("dx") __t3, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
instrument_get_user!(__val_gu);
val = __val_gu as i32;
__v5 = __ret_gu as i64;
__v4 = __v5;
if __v4 != 0 {
return 14i32.wrapping_neg();
}
unsafe {
*((*data).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(i as isize) = val as i64;
}
i = i.wrapping_add(1);
}
} else {
size = get_elem_size(r#type, count);
if size < 0 {
while false {
}
unsafe { _dev_err((*card).dev as *const device, c"snd_ioctl32_ctl_elem_value: unknown type %d\n".as_ptr() as *const u8, r#type) };
return 22i32.wrapping_neg();
}
if copy_from_user((unsafe { (*data).value.bytes.data.as_mut_ptr() as *mut u8 }) as *mut (), valuep as *const (), size as u64) != 0 {
return 14i32.wrapping_neg();
}
}
unsafe {
*(typep as *mut i32) = r#type;
}
unsafe {
*(countp as *mut i32) = count;
}
0
}

fn copy_ctl_value_to_user(userdata: *mut (), valuep: *mut (), data: *mut snd_ctl_elem_value, r#type: i32, count: i32) -> i32 {
let mut size: i32 = 0;
if r#type == 1 || r#type == 2 {
for i in 0..count {
let intp: *mut i32 = valuep as *mut i32;
let val: i32 = (unsafe { *((*data).value.integer.value.as_mut_ptr() as *mut i64).wrapping_offset(i as isize) }) as i32;
let mut __v0: i64;
might_fault();
let mut __v1: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = val;
let __ptr: *mut i32 = intp.wrapping_offset(i as isize);
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t0, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
instrument_put_user!(__x, __ptr);
__v1 = __ret_pu as i64;
__v0 = __v1;
if __v0 != 0 {
return 14i32.wrapping_neg();
}
}
} else {
size = get_elem_size(r#type, count);
if copy_to_user(valuep, (unsafe { (*data).value.bytes.data.as_mut_ptr() as *mut u8 }) as *const (), size as u64) != 0 {
return 14i32.wrapping_neg();
}
}
if copy_to_user((unsafe { core::ptr::addr_of_mut!((*(userdata as *mut snd_ctl_elem_value32)).id) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data).id) }) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 {
return 14i32.wrapping_neg();
}
0
}

fn __ctl_elem_read_user(card: *mut snd_card, userdata: *mut (), valuep: *mut ()) -> i32 {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
let mut r#type: i32 = 0;
let mut count: i32 = 0;
let mut data: *mut snd_ctl_elem_value = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(data) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut snd_ctl_elem_value;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_ctl_elem_value>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut snd_ctl_elem_value;
data = __v1;
if data.is_null() {
return 12i32.wrapping_neg();
}
err = copy_ctl_value_from_user(card, data, userdata, valuep, unsafe { &mut (*core::ptr::addr_of_mut!(r#type)) }, unsafe { &mut (*core::ptr::addr_of_mut!(count)) });
if err < 0 {
return err;
}
err = snd_ctl_elem_read(card, data);
if err < 0 {
return err;
}
copy_ctl_value_to_user(userdata, valuep, data, r#type, count)
}

fn ctl_elem_read_user(card: *mut snd_card, userdata: *mut (), valuep: *mut ()) -> i32 {
let mut err: i32 = unsafe { snd_power_ref_and_wait(card) };
if err < 0 {
return err;
}
err = __ctl_elem_read_user(card, userdata, valuep);
snd_power_unref(card);
err
}

fn __ctl_elem_write_user(file: *mut snd_ctl_file, userdata: *mut (), valuep: *mut ()) -> i32 {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let card: *mut snd_card = unsafe { (*file).card };
let mut err: i32 = 0;
let mut r#type: i32 = 0;
let mut count: i32 = 0;
let mut data: *mut snd_ctl_elem_value = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(data) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut snd_ctl_elem_value;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_ctl_elem_value>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut snd_ctl_elem_value;
data = __v1;
if data.is_null() {
return 12i32.wrapping_neg();
}
err = copy_ctl_value_from_user(card, data, userdata, valuep, unsafe { &mut (*core::ptr::addr_of_mut!(r#type)) }, unsafe { &mut (*core::ptr::addr_of_mut!(count)) });
if err < 0 {
return err;
}
err = snd_ctl_elem_write(card, file, data);
if err < 0 {
return err;
}
copy_ctl_value_to_user(userdata, valuep, data, r#type, count)
}

fn ctl_elem_write_user(file: *mut snd_ctl_file, userdata: *mut (), valuep: *mut ()) -> i32 {
let card: *mut snd_card = unsafe { (*file).card };
let mut err: i32 = unsafe { snd_power_ref_and_wait(card) };
if err < 0 {
return err;
}
err = __ctl_elem_write_user(file, userdata, valuep);
snd_power_unref(card);
err
}

fn snd_ctl_elem_read_user_compat(card: *mut snd_card, data32: *mut snd_ctl_elem_value32) -> i32 {
ctl_elem_read_user(card, data32 as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).value) }) as *mut ())
}

fn snd_ctl_elem_write_user_compat(file: *mut snd_ctl_file, data32: *mut snd_ctl_elem_value32) -> i32 {
ctl_elem_write_user(file, data32 as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).value) }) as *mut ())
}

fn snd_ctl_elem_add_compat(file: *mut snd_ctl_file, mut data32: *mut snd_ctl_elem_info32, replace: i32) -> i32 {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut data: *mut snd_ctl_elem_info = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(data) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_kfree(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut snd_ctl_elem_info;
let __obj_size: u64 = size_mul(core::mem::size_of::<snd_ctl_elem_info>() as u64, 1u64);
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
__v1 = __v2 as *mut snd_ctl_elem_info;
data = __v1;
if data.is_null() {
return 12i32.wrapping_neg();
}
// id, type, access, count
let __v7: bool = copy_from_user((unsafe { core::ptr::addr_of_mut!((*data).id) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).id) }) as *const (), core::mem::size_of::<snd_ctl_elem_id>() as u64) != 0 || copy_from_user((unsafe { core::ptr::addr_of_mut!((*data).r#type) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).r#type) }) as *const (), 3u64.wrapping_mul(core::mem::size_of::<u32>() as u64)) != 0;
if __v7 {
return 14i32.wrapping_neg();
}
let mut __v8: i64;
might_fault();
let mut __v9: i64;
{
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).owner) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
(*data).owner = __val_gu as i32;
}
__v9 = __ret_gu as i64;
}
__v8 = __v9;
if __v8 != 0 {
return 14i32.wrapping_neg();
}
match unsafe { (*data).r#type } {
1 | 2 => {
let mut __v12: i64;
might_fault();
let mut __v13: i64;
{
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t2: u64;
let __t3: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).value.integer.min) as u64 => __t2, lateout("dx") __t3, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
(*data).value.integer.min = (__val_gu as i32) as i64;
}
__v13 = __ret_gu as i64;
}
__v12 = __v13;
let __v16: bool = if __v12 != 0 {
true
} else {
let mut __v17: i64;
might_fault();
let mut __v18: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t4: u64;
let __t5: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).value.integer.max) as u64 => __t4, lateout("dx") __t5, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t4 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t5 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
(*data).value.integer.max = (__val_gu as i32) as i64;
}
__v18 = __ret_gu as i64;
__v17 = __v18;
__v17 != 0
};
let __v21: bool = if __v16 {
true
} else {
let mut __v22: i64;
might_fault();
let mut __v23: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t6: u64;
let __t7: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).value.integer.step) as u64 => __t6, lateout("dx") __t7, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t6 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t7 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
(*data).value.integer.step = (__val_gu as i32) as i64;
}
__v23 = __ret_gu as i64;
__v22 = __v23;
__v22 != 0
};
if __v21 {
return 14i32.wrapping_neg();
}
}
6 => {
if copy_from_user((unsafe { core::ptr::addr_of_mut!((*data).value.integer64) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).value.integer64) }) as *const (), core::mem::size_of::<__SlateRecord_asound_h_1170>() as u64) != 0 {
return 14i32.wrapping_neg();
}
}
3 => {
if copy_from_user((unsafe { core::ptr::addr_of_mut!((*data).value.enumerated) }) as *mut (), (unsafe { core::ptr::addr_of_mut!((*data32).value.enumerated) }) as *const (), core::mem::size_of::<__SlateRecord_asound_h_1175>() as u64) != 0 {
return 14i32.wrapping_neg();
}
unsafe {
(*data).value.enumerated.names_ptr = compat_ptr((*data).value.enumerated.names_ptr as u32) as u64;
}
}
_ => {
}
}
snd_ctl_elem_add(file, data, replace)
}

#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_ioctl_compat")]
extern "C" fn snd_ctl_ioctl_compat(file: *mut file, cmd: u32, arg: u64) -> i64 {
let mut p: *mut snd_kctl_ioctl = unsafe { core::mem::zeroed() };
let argp: *mut () = compat_ptr(arg as u32);
let mut err: i32 = 0;
let ctl: *mut snd_ctl_file = (unsafe { (*file).private_data }) as *mut snd_ctl_file;
if snd_BUG_ON!(ctl.is_null() || unsafe { (*ctl).card }.is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
}
match cmd {
1077957908 | 1077957909 | 2147767552 | 2147767761 | 2172146945 | 3221509398 | 3221509584 | 3221771546..=3221771548 | 3222820098 | 3225441561 => {
return snd_ctl_ioctl(file, cmd, argp as u64);
}
3225965840 => {
return snd_ctl_elem_list_compat(unsafe { (*ctl).card }, argp as *mut snd_ctl_elem_list32) as i64;
}
3239073041 => {
return snd_ctl_elem_info_compat(ctl, argp as *mut snd_ctl_elem_info32) as i64;
}
3267646738 => {
return snd_ctl_elem_read_user_compat(unsafe { (*ctl).card }, argp as *mut snd_ctl_elem_value32) as i64;
}
3267646739 => {
return snd_ctl_elem_write_user_compat(ctl, argp as *mut snd_ctl_elem_value32) as i64;
}
3239073047 => {
return snd_ctl_elem_add_compat(ctl, argp as *mut snd_ctl_elem_info32, 0i32) as i64;
}
3239073048 => {
return snd_ctl_elem_add_compat(ctl, argp as *mut snd_ctl_elem_info32, 1i32) as i64;
}
_ => {
}
}
let mut __UNIQUE_ID_guard_594: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!(snd_ioctl_rwsem) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_594) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_595: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!(snd_ioctl_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_595) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
p = list_first_entry__12!(unsafe { core::ptr::addr_of_mut!(snd_control_compat_ioctls) });
while list_entry_is_head!(p, unsafe { core::ptr::addr_of_mut!(snd_control_compat_ioctls) }, list) == 0 {
if (unsafe { (*p).fioctl }) != None {
err = unsafe { unsafe { (*p).fioctl }.unwrap()((*ctl).card, ctl, cmd, arg) };
if err != 515i32.wrapping_neg() {
return err as i64;
}
}
p = list_next_entry__13!(p, list);
}
515i32.wrapping_neg() as i64
}

// ioctl32 compat
// control layers (audio LED etc.)
/// snd_ctl_request_layer - request to use the layer
/// @module_name: Name of the kernel module (NULL == build-in)
///
/// Return: zero if successful, or an error code when the module cannot be loaded
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_request_layer")]
extern "C" fn snd_ctl_request_layer(module_name: *const u8) -> i32 {
let mut __UNIQUE_ID_unlock_597: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_597) as *mut *mut __SlateRecord_rwsem_h_259;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_259;
let mut lops: core::mem::MaybeUninit<*mut snd_ctl_layer_ops> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut snd_ctl_layer_ops = core::ptr::addr_of_mut!(lops) as *mut *mut snd_ctl_layer_ops;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if module_name.is_null() {
return 0;
} else {
'__join_0: {
core::ptr::write(__slate_slot_1, class_rwsem_read_constructor(core::ptr::addr_of_mut!(snd_ctl_layer_rwsem)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_0.set(true);
if !class_rwsem_read_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_rwsem_read_is_conditional {
if true {
*__slate_slot_2 = snd_ctl_layer;
loop {
if !(*__slate_slot_2).is_null() {
if strcmp((*(*__slate_slot_2)).module_name, module_name) == 0 {
break;
} else {
*__slate_slot_2 = (*(*__slate_slot_2)).next;
}
} else {
break '__join_0;
}
}
return 0;
}
}
}
__slate_cleanup_armed_0.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_rwsem_read_destructor(__slate_slot_1 as *mut _);
return __request_module(true, module_name);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_request_layer_598: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const u8) -> i32>, *mut ()>(Some(snd_ctl_request_layer)) };

/// snd_ctl_register_layer - register new control layer
/// @lops: operation structure
///
/// The new layer can track all control elements and do additional
/// operations on top (like audio LED handling).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_register_layer")]
extern "C" fn snd_ctl_register_layer(lops: *mut snd_ctl_layer_ops) {
let mut __UNIQUE_ID_unlock_602: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_602) as *mut *mut __SlateRecord_rwsem_h_259;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_259;
let mut card: core::mem::MaybeUninit<*mut snd_card> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut snd_card = core::ptr::addr_of_mut!(card) as *mut *mut snd_card;
let mut __UNIQUE_ID_unlock_600: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_600) as *mut *mut __SlateRecord_rwsem_h_270;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_270;
let mut card_number: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(card_number) as *mut i32;
let __slate_cleanup_ptr_6 = __slate_slot_6;
let __slate_cleanup_armed_6 = core::cell::Cell::new(false);
let __slate_cleanup_6 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_6.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_6 as *mut _) };
}
});
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_5 as *mut _) };
}
});
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { __free_snd_card_unref(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_6, class_rwsem_write_constructor(core::ptr::addr_of_mut!(snd_ctl_layer_rwsem)));
__slate_cleanup_armed_6.set(true);
core::ptr::write(__slate_slot_5, (core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_5.set(true);
if !class_rwsem_write_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6))).is_null() || !class_rwsem_write_is_conditional {
if true {
(*lops).next = snd_ctl_layer;
snd_ctl_layer = lops;
}
}
__slate_cleanup_armed_5.set(false);
__class_rwsem_write_cleanup_ctx(__slate_slot_5 as *mut _);
__slate_cleanup_armed_6.set(false);
class_rwsem_write_destructor(__slate_slot_6 as *mut _);
*__slate_slot_7 = 0;
while *__slate_slot_7 < 8 {
core::ptr::write(__slate_slot_4, snd_card_ref(*__slate_slot_7));
__slate_cleanup_armed_4.set(true);
if !(*__slate_slot_4).is_null() {
core::ptr::write(__slate_slot_3, class_rwsem_read_constructor(core::ptr::addr_of_mut!((*(*__slate_slot_4)).controls_rwsem)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!((*(*__slate_slot_4)).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_2.set(true);
if !class_rwsem_read_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_rwsem_read_is_conditional {
if true {
unsafe { (*lops).lregister }.unwrap()(*__slate_slot_4);
}
}
__slate_cleanup_armed_2.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_rwsem_read_destructor(__slate_slot_3 as *mut _);
}
__slate_cleanup_armed_4.set(false);
__free_snd_card_unref(__slate_slot_4 as *mut _);
let __v1: i32 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_add(1);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_register_layer_603: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_ctl_layer_ops)>, *mut ()>(Some(snd_ctl_register_layer)) };

/// snd_ctl_disconnect_layer - disconnect control layer
/// @lops: operation structure
///
/// It is expected that the information about tracked cards
/// is freed before this call (the disconnect callback is
/// not called here).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_disconnect_layer")]
extern "C" fn snd_ctl_disconnect_layer(lops: *mut snd_ctl_layer_ops) {
let mut lops2: *mut snd_ctl_layer_ops = unsafe { core::mem::zeroed() };
let mut prev_lops2: *mut snd_ctl_layer_ops = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_604: __SlateRecord_rwsem_h_270 = class_rwsem_write_constructor(unsafe { core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_604) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_write_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_605: *mut __SlateRecord_rwsem_h_270 = ((unsafe { core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_270;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_605) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
lops2 = unsafe { snd_ctl_layer };
prev_lops2 = core::ptr::null_mut::<snd_ctl_layer_ops>();
while !lops2.is_null() {
if lops2 == lops {
if prev_lops2.is_null() {
unsafe {
snd_ctl_layer = (*lops).next;
}
} else {
unsafe {
(*prev_lops2).next = (*lops).next;
}
}
break;
}
prev_lops2 = lops2;
lops2 = unsafe { (*lops2).next };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_disconnect_layer_606: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_ctl_layer_ops)>, *mut ()>(Some(snd_ctl_disconnect_layer)) };

// INIT PART
static snd_ctl_f_ops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.read = Some(snd_ctl_read);
__t0.poll = Some(snd_ctl_poll);
__t0.unlocked_ioctl = Some(snd_ctl_ioctl);
__t0.compat_ioctl = Some(snd_ctl_ioctl_compat);
__t0.open = Some(snd_ctl_open);
__t0.release = Some(snd_ctl_release);
__t0.fasync = Some(snd_ctl_fasync);
__t0
});

// call lops under rwsems; called from snd_ctl_dev_*() below()
/// registration of the control device
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_dev_register")]
extern "C" fn snd_ctl_dev_register(device: *mut snd_device) -> i32 {
let card: *mut snd_card = (unsafe { (*device).device_data }) as *mut snd_card;
let err: i32 = unsafe { snd_register_device(0i32, card, 1i32.wrapping_neg(), core::ptr::addr_of!(snd_ctl_f_ops.0), card as *mut (), (*card).ctl_dev) };
if err < 0 {
return err;
}
loop {
{
let mut lops: *mut snd_ctl_layer_ops = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_607: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_607) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_608: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*card).controls_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_608) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_guard_609: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_609) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_610: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_610) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
lops = unsafe { snd_ctl_layer };
while !lops.is_null() {
unsafe { unsafe { (*lops).lregister }.unwrap()(card) };
lops = unsafe { (*lops).next };
}
}
if true {
break;
}
}
0
}

/// disconnection of the control device
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_dev_disconnect")]
extern "C" fn snd_ctl_dev_disconnect(device: *mut snd_device) -> i32 {
let mut __UNIQUE_ID_unlock_616: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_616) as *mut *mut __SlateRecord_rwsem_h_259;
let mut __UNIQUE_ID_guard_615: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_615) as *mut __SlateRecord_rwsem_h_259;
let mut __UNIQUE_ID_unlock_614: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_614) as *mut *mut __SlateRecord_rwsem_h_259;
let mut __UNIQUE_ID_guard_613: core::mem::MaybeUninit<__SlateRecord_rwsem_h_259> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_613) as *mut __SlateRecord_rwsem_h_259;
let mut lops: core::mem::MaybeUninit<*mut snd_ctl_layer_ops> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut snd_ctl_layer_ops = core::ptr::addr_of_mut!(lops) as *mut *mut snd_ctl_layer_ops;
let mut __UNIQUE_ID_unlock_612: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_670> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut __SlateRecord_spinlock_h_670 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_612) as *mut *mut __SlateRecord_spinlock_h_670;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_670> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_spinlock_h_670 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_670;
let mut ctl: core::mem::MaybeUninit<*mut snd_ctl_file> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut snd_ctl_file = core::ptr::addr_of_mut!(ctl) as *mut *mut snd_ctl_file;
let mut card: *mut snd_card = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_6 = __slate_slot_6;
let __slate_cleanup_armed_6 = core::cell::Cell::new(false);
let __slate_cleanup_6 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_6.get() {
unsafe { class_read_lock_irqsave_destructor(__slate_cleanup_ptr_6 as *mut _) };
}
});
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { __class_read_lock_irqsave_cleanup_ctx(__slate_cleanup_ptr_5 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_rwsem_read_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
{
card = (*device).device_data as *mut snd_card;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_6, class_read_lock_irqsave_constructor(core::ptr::addr_of_mut!((*card).controls_rwlock)));
__slate_cleanup_armed_6.set(true);
core::ptr::write(__slate_slot_5, (core::ptr::addr_of_mut!((*card).controls_rwlock) as u64) as *mut __SlateRecord_spinlock_h_670);
__slate_cleanup_armed_5.set(true);
if !class_read_lock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_6))).is_null() || !class_read_lock_irqsave_is_conditional {
if true {
*__slate_slot_7 = list_first_entry__11!(core::ptr::addr_of_mut!((*card).ctl_files));
while list_entry_is_head!(*__slate_slot_7, core::ptr::addr_of_mut!((*card).ctl_files), list) == 0 {
wake_up!(core::ptr::addr_of_mut!((*(*__slate_slot_7)).change_sleep));
snd_kill_fasync((*(*__slate_slot_7)).fasync, 29i32, 4i32);
*__slate_slot_7 = list_next_entry__14!(*__slate_slot_7, list);
}
}
}
}
__slate_cleanup_armed_5.set(false);
__class_read_lock_irqsave_cleanup_ctx(__slate_slot_5 as *mut _);
__slate_cleanup_armed_6.set(false);
class_read_lock_irqsave_destructor(__slate_slot_6 as *mut _);
loop {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_3, class_rwsem_read_constructor(core::ptr::addr_of_mut!((*card).controls_rwsem)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!((*card).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_2.set(true);
core::ptr::write(__slate_slot_1, class_rwsem_read_constructor(core::ptr::addr_of_mut!(snd_ctl_layer_rwsem)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(snd_ctl_layer_rwsem) as u64) as *mut __SlateRecord_rwsem_h_259);
__slate_cleanup_armed_0.set(true);
*__slate_slot_4 = snd_ctl_layer;
while !(*__slate_slot_4).is_null() {
unsafe { (*(*__slate_slot_4)).ldisconnect }.unwrap()(card);
*__slate_slot_4 = (*(*__slate_slot_4)).next;
}
__slate_cleanup_armed_0.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_rwsem_read_destructor(__slate_slot_1 as *mut _);
__slate_cleanup_armed_2.set(false);
__class_rwsem_read_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_rwsem_read_destructor(__slate_slot_3 as *mut _);
if true {
break;
}
}
return snd_unregister_device((*card).ctl_dev);
}
}

/// free all controls
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_dev_free")]
extern "C" fn snd_ctl_dev_free(device: *mut snd_device) -> i32 {
let mut __UNIQUE_ID_unlock_618: core::mem::MaybeUninit<*mut __SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_618) as *mut *mut __SlateRecord_rwsem_h_270;
let mut scope: core::mem::MaybeUninit<__SlateRecord_rwsem_h_270> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_rwsem_h_270 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rwsem_h_270;
let mut control: core::mem::MaybeUninit<*mut snd_kcontrol> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut snd_kcontrol = core::ptr::addr_of_mut!(control) as *mut *mut snd_kcontrol;
let mut card: *mut snd_card = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_rwsem_write_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_rwsem_write_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
card = (*device).device_data as *mut snd_card;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_1, class_rwsem_write_constructor(core::ptr::addr_of_mut!((*card).controls_rwsem)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*card).controls_rwsem) as u64) as *mut __SlateRecord_rwsem_h_270);
__slate_cleanup_armed_0.set(true);
if !class_rwsem_write_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_rwsem_write_is_conditional {
if true {
while list_empty(core::ptr::addr_of_mut!((*card).controls) as *const list_head) == 0 {
*__slate_slot_2 = snd_kcontrol!(unsafe { (*card).controls.next });
__snd_ctl_remove(card, *__slate_slot_2, false);
}
xa_destroy(core::ptr::addr_of_mut!((*card).ctl_numids));
xa_destroy(core::ptr::addr_of_mut!((*card).ctl_hash));
}
}
__slate_cleanup_armed_0.set(false);
__class_rwsem_write_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_rwsem_write_destructor(__slate_slot_1 as *mut _);
put_device((*card).ctl_dev);
return 0;
}
}

/// create control core:
/// called from init.c
#[unsafe(no_mangle)]
extern "C" fn snd_ctl_create(card: *mut snd_card) -> i32 {
let mut err: i32 = 0;
if snd_BUG_ON!(card.is_null()) != 0 {
return 6i32.wrapping_neg();
}
if snd_BUG_ON!((unsafe { (*card).number }) < 0 || (unsafe { (*card).number }) >= 8) != 0 {
return 6i32.wrapping_neg();
}
err = unsafe { snd_device_alloc(core::ptr::addr_of_mut!((*card).ctl_dev), card) };
if err < 0 {
return err;
}
unsafe { dev_set_name((*card).ctl_dev, c"controlC%d".as_ptr() as *const u8, (*card).number) };
err = unsafe { snd_device_new(card, 11u32, card as *mut (), core::ptr::addr_of!(ops.0)) };
if err < 0 {
unsafe { put_device((*card).ctl_dev) };
}
err
}

static ops: __SlateConst<snd_device_ops> = __SlateConst(snd_device_ops { dev_free: Some(snd_ctl_dev_free), dev_register: Some(snd_ctl_dev_register), dev_disconnect: Some(snd_ctl_dev_disconnect) });

// Frequently used control callbacks/helpers
/// snd_ctl_boolean_mono_info - Helper function for a standard boolean info
/// callback with a mono channel
/// @kcontrol: the kcontrol instance
/// @uinfo: info to store
///
/// This is a function that can be used as info callback for a standard
/// boolean control with a single mono channel.
///
/// Return: Zero (always successful)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_boolean_mono_info")]
extern "C" fn snd_ctl_boolean_mono_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
unsafe {
(*uinfo).r#type = 1;
}
unsafe {
(*uinfo).count = 1;
}
unsafe {
(*uinfo).value.integer.min = 0;
}
unsafe {
(*uinfo).value.integer.max = 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_boolean_mono_info_619: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>, *mut ()>(Some(snd_ctl_boolean_mono_info)) };

/// snd_ctl_boolean_stereo_info - Helper function for a standard boolean info
/// callback with stereo two channels
/// @kcontrol: the kcontrol instance
/// @uinfo: info to store
///
/// This is a function that can be used as info callback for a standard
/// boolean control with stereo two channels.
///
/// Return: Zero (always successful)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_boolean_stereo_info")]
extern "C" fn snd_ctl_boolean_stereo_info(kcontrol: *mut snd_kcontrol, uinfo: *mut snd_ctl_elem_info) -> i32 {
unsafe {
(*uinfo).r#type = 1;
}
unsafe {
(*uinfo).count = 2;
}
unsafe {
(*uinfo).value.integer.min = 0;
}
unsafe {
(*uinfo).value.integer.max = 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_boolean_stereo_info_620: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_kcontrol, *mut snd_ctl_elem_info) -> i32>, *mut ()>(Some(snd_ctl_boolean_stereo_info)) };

/// snd_ctl_enum_info - fills the info structure for an enumerated control
/// @info: the structure to be filled
/// @channels: the number of the control's channels; often one
/// @items: the number of control values; also the size of @names
/// @names: an array containing the names of all control values
///
/// Sets all required fields in @info to their appropriate values.
/// If the control's accessibility is not the default (readable and writable),
/// the caller has to fill @info->access.
///
/// Return: Zero (always successful)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.control.snd_ctl_enum_info")]
extern "C" fn snd_ctl_enum_info(info: *mut snd_ctl_elem_info, channels: u32, items: u32, names: *const *const u8) -> i32 {
unsafe {
(*info).r#type = 3;
}
unsafe {
(*info).count = channels;
}
unsafe {
(*info).value.enumerated.items = items;
}
if items == 0 {
return 0;
}
if (unsafe { (*info).value.enumerated.item }) >= items {
unsafe {
(*info).value.enumerated.item = items.wrapping_sub(1);
}
}
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { strlen(*names.wrapping_offset((*info).value.enumerated.item as isize)) }) >= (core::mem::size_of::<[u8; 64]>() as u64)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"ALSA: too long item name '%s'\n".as_ptr() as *const u8, *names.wrapping_offset((*info).value.enumerated.item as isize)) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_snd_ctl_enum_info_3, sym __slate_asm_string_snd_ctl_enum_info_4, const 2579 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, *names.wrapping_offset((*info).value.enumerated.item as isize)) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { sized_strscpy((*info).value.enumerated.name.as_mut_ptr() as *mut u8, *names.wrapping_offset((*info).value.enumerated.item as isize), (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__42!() as u64).wrapping_add(__must_be_cstr__4!() as u64)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_621: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_ctl_enum_info_622: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_ctl_elem_info, u32, u32, *const *const u8) -> i32>, *mut ()>(Some(snd_ctl_enum_info)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_snd_ctl_enum_info_3: [u8; 31] = *b"ALSA: too long item name '%s'\n\0";

static __slate_asm_string_snd_ctl_enum_info_4: [u8; 50] = *b"/home/takashi/c-corpus/linux/sound/core/control.c\0";
