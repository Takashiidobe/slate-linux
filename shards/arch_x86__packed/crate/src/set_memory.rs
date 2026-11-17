
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
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! Copyright 2002 Andi Kleen, SuSE Labs.
//! Thanks to Ben LaHaise for precious feedback.
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! Internal kernel timer frequency
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
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
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! This counts to 15. Any more, it will return 16th argument.
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Prevent speculative execution past this barrier.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! writeable
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! Helper macro for scoped_guard().
//!
//! Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
//! compiler would be sure that for the unconditional locks the body of the
//! loop (caller-provided code glued to the else clause) could not be skipped.
//! It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
//! hard to deduce (even if could be proven true for unconditional locks).
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! @a is a power of 2 value
//! strscpy - Copy a C-string into a sized buffer
//! @dst: Where to copy the string to
//! @src: Where to copy the string from
//! @...: Size of destination buffer (optional)
//!
//! Copy the source string @src, or as much of it as fits, into the
//! destination @dst buffer. The behavior is undefined if the string
//! buffers overlap. The destination @dst buffer is always NUL terminated,
//! unless it's zero-sized.
//!
//! The size argument @... is only required when @dst is not an array, or
//! when the copy needs to be smaller than sizeof(@dst).
//!
//! Preferred to strncpy() since it always returns a valid string, and
//! doesn't unnecessarily force the tail of the destination buffer to be
//! zero padded. If padding is desired please use strscpy_pad().
//!
//! Returns the number of characters copied in @dst (not including the
//! trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
//! truncated.
//! __sign_use for integer expressions:
//!   bit #0 set if ok for unsigned comparisons
//!   bit #1 set if ok for signed comparisons
//!
//! In particular, statically non-negative signed integer expressions
//! are ok for both.
//!
//! NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
//! in expressions, and are accepted for signed conversions for now.
//! This is debatable.
//!
//! Note that 'x' is the original expression, and 'ux' is the unique variable
//! that contains the value.
//!
//! We use 'ux' for pure type checking, and 'x' for when we need to look at the
//! value (but without evaluating it for side effects!
//! Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
//!
//! Pointers end up being checked by the normal C type rules at the actual
//! comparison, and these expressions only need to be careful to not cause
//! warnings for pointer use.
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
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Defines VMX CPU feature bits
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! wait_queue_entry::flags
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
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! MXCSR will be initialized
//! Macros to add or remove encryption attribute
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
//! kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
//! kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
//! and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
//! per-IO O_SYNC
//! File supports atomic writes
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! See the comment in asm-generic/io.h about ioremap_np().
//! 8237 DMA controllers
//! We are using top down, so it is safe to use 0 here
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
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
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! ***DANGEROUS*** don't add guard page
//! Monochrome Text Display
//! number of edd_info structs starting at EDDBUF
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Various well-known legacy memory ranges in physical memory:
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
// __acquire() - function to acquire context lock exclusively
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x exclusively.
// __release() - function to release context lock exclusively
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
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
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
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
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
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
// check_add_overflow() - Calculate addition with overflow checking
// @a: first addend
// @b: second addend
// @d: pointer to store sum
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted addition, regardless of whether
// wrap-around occurred.
// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
// check_mul_overflow() - Calculate multiplication with overflow checking
// @a: first factor
// @b: second factor
// @d: pointer to store product
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted multiplication, regardless of whether
// wrap-around occurred.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// strlen("ratelimit") + 1
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "clflush" CLFLUSH instruction
// "sse"
// "pdpe1gb" GB pages
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "hypervisor" Running on a hypervisor
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// No such file or directory
// I/O error
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
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
// N 32-bit words worth of info
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
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
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
// 6 because 6 should be plenty and struct tlb_state will fit in two cache
// lines.
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
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// Set the IDR flag and the IDR_FREE tag
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// Generate SD flag bits
// Implementation specific max size
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// The PCI interface treats multi-function devices as independent
// devices.  The slot/function address of each device is encoded
// in a single byte as follows:
//
// 	7:3 = slot
// 	2:0 = function
//
// PCI_DEVFN(), PCI_SLOT(), and PCI_FUNC() are defined in uapi/linux/pci.h.
// In the interest of not exposing interfaces to user-space unnecessarily,
// the following kernel-only defines are being added here.
// Error values that may be returned by PCI functions
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
// max number of signatures to store
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
/// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
///
/// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
///
/// Assert that there are no concurrent writes to a subset of bits in @var;
/// concurrent readers are permitted. This assertion captures more detailed
/// bit-level properties, compared to the other (word granularity) assertions.
/// Only the bits set in @mask are checked for concurrent modifications, while
/// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
/// are ignored.
///
/// Use this for variables, where some bits must not be modified concurrently,
/// yet other bits are expected to be modified concurrently.
///
/// For example, variables where, after initialization, some bits are read-only,
/// but other bits may still be modified concurrently. A reader may wish to
/// assert that this is true as follows:
///
/// .. code-block:: c
///
/// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
/// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
///
/// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
/// to access the masked bits only, and KCSAN optimistically assumes it is
/// therefore safe, even in the presence of data races, and marking it with
/// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
/// it may still be advisable to do so, since we cannot reason about all compiler
/// optimizations when it comes to bit manipulations (on the reader and writer
/// side). If you are sure nothing can go wrong, we can write the above simply
/// as:
///
/// .. code-block:: c
///
/// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
/// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
///
/// Another example, where this may be used, is when certain bits of @var may
/// only be modified when holding the appropriate lock, but other bits may still
/// be modified concurrently. Writers, where other bits may change concurrently,
/// could use the assertion as follows:
///
/// .. code-block:: c
///
/// 	spin_lock(&foo_lock);
/// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
/// 	old_flags = flags;
/// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
/// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
/// 	spin_unlock(&foo_lock);
///
/// @var: variable to assert on
/// @mask: only check for modifications to bits set in @mask
macro_rules! ASSERT_EXCLUSIVE_BITS {
($var:expr, $mask:expr) => {
loop {
kcsan_set_access_mask($mask);
__kcsan_check_access((unsafe { core::ptr::addr_of!($var) }) as *const (), core::mem::size_of::<u64>() as u64, 8i32);
kcsan_set_access_mask(0u64);
kcsan_atomic_next(1i32);
if true {
break;
}
}
};
}

/// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
///
/// Assert that there are no concurrent writes to @var; other readers are
/// allowed. This assertion can be used to specify properties of concurrent code,
/// where violation cannot be detected as a normal data race.
///
/// For example, if we only have a single writer, but multiple concurrent
/// readers, to avoid data races, all these accesses must be marked; even
/// concurrent marked writes racing with the single writer are bugs.
/// Unfortunately, due to being marked, they are no longer data races. For cases
/// like these, we can use the macro as follows:
///
/// .. code-block:: c
///
/// 	void writer(void) {
/// 		spin_lock(&update_foo_lock);
/// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
/// 		WRITE_ONCE(shared_foo, ...);
/// 		spin_unlock(&update_foo_lock);
/// 	}
/// 	void reader(void) {
/// 		// update_foo_lock does not need to be held!
/// 		... = READ_ONCE(shared_foo);
/// 	}
///
/// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
/// checking if a clear scope where no concurrent writes are expected exists.
///
/// @var: variable to assert on
macro_rules! ASSERT_EXCLUSIVE_WRITER {
($var:expr) => {
__kcsan_check_access((unsafe { core::ptr::addr_of_mut!($var) }) as *const (), core::mem::size_of::<seqcount>() as u64, 8i32);
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

macro_rules! BIT_WORD__2 {
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

/// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
/// expression but avoids the generation of any code, even if that expression
/// has side-effects.
macro_rules! BUILD_BUG_ON_INVALID {
() => {
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
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! NODE_DATA {
($nid:expr) => {
unsafe { *(core::ptr::addr_of_mut!(node_data) as *mut *mut pglist_data).wrapping_offset(($nid) as isize) }
};
}

macro_rules! PFN_DOWN {
($x:expr) => {
($x) >> 12i32
};
}

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! REQUIRED_MASK_BIT_SET {
($x:expr) => {
(if ($x) >> 5i32 == 0 { 117481833 } else { if ($x) >> 5i32 == 1 { 536870912 } else { if ($x) >> 5i32 == 3 { 36732928 } else { 0 } } }) & 1u32 << (($x) & 31)
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VIRTUAL_BUG_ON {
() => {
while false {
}
};
}

macro_rules! VM_BUG_ON_FOLIO {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON_PAGE {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON_PGFLAGS {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__2 {
() => {
18446744071562067968u64
};
}

macro_rules! _AC__3 {
() => {
16045481047390945280u64
};
}

macro_rules! _AT {
($T:ty, $X:expr) => {
($X) as $T
};
}

macro_rules! _BITUL {
($x:expr) => {
_UL!() << $x
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__7 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_types_h_295, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__9 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_64_types_h_21, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__8 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_types_h_368, $val) };
if true {
break;
}
}
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

macro_rules! __WRITE_ONCE__4 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_types_h_342, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_64_types_h_22, $val) };
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

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__4 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__13 {
() => {
((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__16 {
() => {
((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__12 {
() => {
((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __p4d {
($x:expr) => {
native_make_p4d($x)
};
}

macro_rules! __pa {
($x:expr) => {
__phys_addr!(($x) as u64)
};
}

/// __pa_symbol should be used for C visible symbols.
/// This seems to be the official gcc blessed way to do such arithmetic.
///
/// We need __phys_reloc_hide() here because gcc may assume that there is no
/// overflow during __pa() calculation and can optimize it unexpectedly.
/// Newer versions of gcc provide -fno-strict-overflow switch to handle this
/// case properly. Once all supported versions of gcc understand it, we can
/// remove this Voodoo magic stuff. (i.e. once gcc3.x is deprecated)
macro_rules! __pa_symbol {
($x:expr) => {
__phys_addr_symbol(__phys_reloc_hide!(($x) as u64))
};
}

macro_rules! __pgd {
($x:expr) => {
native_make_pgd($x)
};
}

macro_rules! __phys_addr {
($x:expr) => {
__phys_addr_nodebug($x)
};
}

macro_rules! __phys_reloc_hide {
($x:expr) => {
$x
};
}

macro_rules! __pmd {
($x:expr) => {
native_make_pmd($x)
};
}

macro_rules! __pte {
($x:expr) => {
native_make_pte($x)
};
}

macro_rules! __pte__2 {
($x:expr) => {
native_make_pte(($x) as u64)
};
}

macro_rules! __pud {
($x:expr) => {
native_make_pud($x)
};
}

macro_rules! __va {
($x:expr) => {
(($x) as u64).wrapping_add(unsafe { page_offset_base }) as *mut ()
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
macro_rules! __verify_pcpu_ptr__4 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u64>() as *const ();
if true {
break;
}
}
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
macro_rules! __verify_pcpu_ptr__3 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

macro_rules! _static_cpu_has__6 {
($bit:expr) => {
__static_cpu_has__const5(($bit) as u16) as i32
};
}

macro_rules! _static_cpu_has__8 {
($bit:expr) => {
__static_cpu_has__const8(($bit) as u16) as i32
};
}

macro_rules! _static_cpu_has__5 {
($bit:expr) => {
__static_cpu_has__const6(($bit) as u16) as i32
};
}

macro_rules! _static_cpu_has__4 {
($bit:expr) => {
__static_cpu_has__const4(($bit) as u16) as i32
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_hooks__4 {
($_do_alloc:expr) => {
{
let mut __v: *mut __SlateRecord_pgtable_64_types_h_21;
let mut __v2: *mut __SlateRecord_pgtable_64_types_h_21;
let mut __v3: *mut __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_hooks__5 {
($_do_alloc:expr) => {
{
let mut __v: *mut __SlateRecord_pgtable_64_types_h_22;
let mut __v2: *mut __SlateRecord_pgtable_64_types_h_22;
let mut __v3: *mut __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_hooks__6 {
($_do_alloc:expr) => {
{
let mut __v: *mut __SlateRecord_pgtable_types_h_368;
let mut __v2: *mut __SlateRecord_pgtable_types_h_368;
let mut __v3: *mut __SlateRecord_pgtable_types_h_368 = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_hooks__8 {
($_do_alloc:expr) => {
{
let mut __v: *mut __SlateRecord_pgtable_types_h_342;
let mut __v2: *mut __SlateRecord_pgtable_types_h_342;
let mut __v3: *mut __SlateRecord_pgtable_types_h_342 = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! boot_cpu_has {
($bit:expr) => {
cpu_has!(unsafe { core::ptr::addr_of_mut!(boot_cpu_data) }, $bit)
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__18 {
($ptr:expr) => {
{
let mut __v: *mut ptdesc;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!() as isize)) }) as *mut ()) as *mut ptdesc;
__v
}
};
}

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled__5 {
($bit:expr) => {
_static_cpu_has__5!($bit)
};
}

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled__7 {
($bit:expr) => {
_static_cpu_has__6!($bit)
};
}

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled {
($bit:expr) => {
_static_cpu_has__4!($bit)
};
}

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled__6 {
($bit:expr) => {
_static_cpu_has__8!($bit)
};
}

macro_rules! cpu_has {
($c:expr, $bit:expr) => {
{
let __v: i32 = if true && REQUIRED_MASK_BIT_SET!($bit) != 0 {
1
} else {
test_cpu_cap!($c, $bit) as i32
};
__v
}
};
}

/// data_race - mark an expression as containing intentional data races
///
/// This data_race() macro is useful for situations in which data races
/// should be forgiven.  One example is diagnostic code that accesses
/// shared variables but is not a part of the core synchronization design.
/// For example, if accesses to a given variable are protected by a lock,
/// except for diagnostic code, then the accesses under the lock should
/// be plain C-language accesses and those in the diagnostic code should
/// use data_race().  This way, KCSAN will complain if buggy lockless
/// accesses to that variable are introduced, even if the buggy accesses
/// are protected by READ_ONCE() or WRITE_ONCE().
///
/// This macro *does not* affect normal code generation, but is a hint
/// to tooling that data races here are to be ignored.  If the access must
/// be atomic *and* KCSAN should ignore the access, use both data_race()
/// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
macro_rules! data_race {
($expr:expr) => {
{
let mut __v: u32;
__kcsan_disable_current();
let __v2: u32 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2__2 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
};
}

macro_rules! irqs_disabled {
() => {
{
let mut __v: i32;
let mut __v2: u64 = 0;
raw_local_save_flags!(__v2);
let mut __v3: i32;
typecheck__2!(u64);
__v3 = arch_irqs_disabled_flags(__v2);
__v = __v3;
__v
}
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! kcsan_wmb {
() => {
while false {
}
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__7 {
($ptr:expr) => {
container_of__18!($ptr)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__2 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).__slate_anon_1.$member) }) as *const list_head, ($head) as *const list_head)
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
list_entry__7!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__11 {
($pos:expr, $member:ident) => {
list_entry__7!(unsafe { (*$pos).__slate_anon_1.$member.next })
};
}

macro_rules! lockdep_assert_held_write {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__4!($type, $x, $y)
};
}

macro_rules! offsetof__15 {
() => {
8u64
};
}

macro_rules! p4d_leaf {
() => {
0i32
};
}

macro_rules! p4d_val {
($x:expr) => {
native_p4d_val($x)
};
}

macro_rules! page_address__2 {
($page:expr) => {
lowmem_page_address($page)
};
}

macro_rules! page_address {
($page:expr) => {
lowmem_page_address(($page) as *const page)
};
}

macro_rules! page_ptdesc {
($p:expr) => {
($p) as *mut ptdesc
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
};
}

/// Must be a compile-time constant, so implement it as a macro
macro_rules! pgd_index {
($a:expr) => {
($a) >> unsafe { pgdir_shift } & 511u64
};
}

macro_rules! pgd_offset {
($mm:expr, $address:expr) => {
pgd_offset_pgd(unsafe { (*$mm).__slate_anon_0.pgd }, $address)
};
}

/// a shortcut which implies the use of the kernel's pgd, instead
/// of a process's
macro_rules! pgd_offset_k {
($address:expr) => {
pgd_offset!(unsafe { core::ptr::addr_of_mut!(init_mm) }, $address)
};
}

macro_rules! pgd_val {
($x:expr) => {
native_pgd_val($x)
};
}

macro_rules! pgprot_val__2 {
($x:expr) => {
unsafe { $x.pgprot }
};
}

macro_rules! pgprot_val {
($x:expr) => {
$x.pgprot
};
}

macro_rules! pgtable_l5_enabled__3 {
() => {
cpu_feature_enabled__7!(528i32)
};
}

macro_rules! pmd_clear {
($pmd:expr) => {
native_pmd_clear($pmd);
};
}

macro_rules! pmd_val {
($x:expr) => {
native_pmd_val($x)
};
}

macro_rules! ptdesc_folio {
($pt:expr) => {
($pt) as *mut folio
};
}

macro_rules! ptdesc_folio__2 {
($pt:expr) => {
($pt) as *const folio
};
}

macro_rules! ptdesc_page {
($pt:expr) => {
($pt) as *mut page
};
}

macro_rules! pte_val {
($x:expr) => {
native_pte_val($x)
};
}

macro_rules! pud_clear {
($pud:expr) => {
native_pud_clear($pud);
};
}

macro_rules! pud_val {
($x:expr) => {
native_pud_val($x)
};
}

macro_rules! raw_local_save_flags {
($flags:expr) => {
loop {
typecheck__2!(u64);
$flags = arch_local_save_flags();
if true {
break;
}
}
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

/// roundup - round up to the next specified multiple
/// @x: the value to up
/// @y: multiple to round up to
///
/// Rounds @x up to next multiple of @y. If @y will always be a power
/// of 2, consider using the faster round_up().
macro_rules! roundup {
($x:expr, $y:expr) => {
{
let mut __v: u64;
let __v2: u64 = $y;
__v = ($x.wrapping_add(__v2.wrapping_sub(1)) / __v2).wrapping_mul(__v2);
__v
}
};
}

macro_rules! set_p4d {
($p4dp:expr, $p4d:expr) => {
native_set_p4d($p4dp, $p4d);
};
}

macro_rules! set_pgd {
($pgdp:expr, $pgd:expr) => {
native_set_pgd($pgdp, $pgd);
};
}

macro_rules! set_pmd {
($pmdp:expr, $pmd:expr) => {
native_set_pmd($pmdp, $pmd);
};
}

macro_rules! set_pte {
($ptep:expr, $pte:expr) => {
native_set_pte($ptep, $pte);
};
}

macro_rules! set_pte_atomic {
($ptep:expr, $pte:expr) => {
native_set_pte_atomic($ptep, $pte);
};
}

macro_rules! set_pud {
($pudp:expr, $pud:expr) => {
native_set_pud($pudp, $pud);
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

macro_rules! static_branch_unlikely__3 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__2 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const2(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! test_bit__6 {
($nr:expr, $addr:expr) => {
{
let __v: bool = true && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! test_bit__4 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_cpu_cap {
($c:expr, $bit:expr) => {
arch_test_bit(($bit) as u64, ((unsafe { (*$c).__slate_anon_9.x86_capability.as_mut_ptr() as *mut u32 }) as *mut u64) as *const u64)
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__2 {
($type:ty) => {
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! virt_addr_valid {
($kaddr:expr) => {
unsafe { __virt_addr_valid(($kaddr) as u64) }
};
}

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut oops_in_progress: i32;
    static mut kmsg_fops: file_operations;
    static mut max_pfn_mapped: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut pgdir_shift: u32;
    static mut ptrs_per_p4d: u32;
    static mut __supported_pte_mask: u64;
    static mut early_boot_irqs_disabled: bool;
    static mut system_state: u32;
    static mut node_data: [*mut pglist_data; 0];
    static mut __cpu_online_mask: cpumask;
    static mut __num_online_cpus: __SlateRecord_types_h_190;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut init_mm: mm_struct;
    static mut x86_platform: x86_platform_ops;
    static mut __tracepoint_mmap_lock_start_locking: tracepoint;
    static mut __tracepoint_mmap_lock_acquire_returned: tracepoint;
    static mut __tracepoint_mmap_lock_released: tracepoint;
    static mut pgd_lock: spinlock;
    static mut pgd_list: list_head;
    static mut direct_gbpages: i32;
    static mut vm_event_states: vm_event_state;
    static mut _debug_pagealloc_enabled_early: bool;
    static mut _text: [u8; 0];
    static mut _stext: [u8; 0];
    static mut _etext: [u8; 0];
    static mut __start_rodata: [u8; 0];
    static mut __end_rodata: [u8; 0];
    static mut __end_rodata_hpage_align: [u8; 0];
    static mut _brk_start: u64;
    static mut _brk_end: u64;
    static mut kernel_set_to_readonly: i32;
    static mut tlb_single_page_flush_ceiling: u64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn pfn_range_is_mapped(arg0: u64, arg1: u64) -> bool;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __virt_addr_valid(arg0: u64) -> bool;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn cachemode2protval(arg0: u32) -> u64;
    fn ____wrong_branch_error() -> bool;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn on_each_cpu_cond_mask(arg0: Option<unsafe extern "C" fn(i32, *mut ()) -> bool>, arg1: Option<unsafe extern "C" fn(*mut ())>, arg2: *mut (), arg3: bool, arg4: *const cpumask);
    fn wbinvd_on_all_cpus();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn down_read(arg0: *mut rw_semaphore);
    fn down_read_trylock(arg0: *mut rw_semaphore) -> i32;
    fn down_write(arg0: *mut rw_semaphore);
    fn down_write_trylock(arg0: *mut rw_semaphore) -> i32;
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn alloc_pages_noprof(arg0: u32, arg1: u32) -> *mut page;
    fn __free_pages(arg0: *mut page, arg1: u32);
    fn __mmap_lock_do_trace_start_locking(arg0: *mut mm_struct, arg1: bool);
    fn __mmap_lock_do_trace_acquire_returned(arg0: *mut mm_struct, arg1: bool, arg2: bool);
    fn __mmap_lock_do_trace_released(arg0: *mut mm_struct, arg1: bool);
    fn __pti_set_user_pgtbl(arg0: *mut __SlateRecord_pgtable_types_h_295, arg1: __SlateRecord_pgtable_types_h_295) -> __SlateRecord_pgtable_types_h_295;
    fn __compiletime_assert_290();
    fn __compiletime_assert_291();
    fn __compiletime_assert_292();
    fn __compiletime_assert_293();
    fn __compiletime_assert_294();
    fn __compiletime_assert_295();
    fn __flush_tlb_all();
    fn flush_tlb_one_kernel(arg0: u64);
    fn flush_tlb_all();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn mod_node_page_state(arg0: *mut pglist_data, arg1: u32, arg2: i64);
    fn pagetable_free_kernel(arg0: *mut ptdesc);
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn memtype_reserve(arg0: u64, arg1: u64, arg2: u32, arg3: *mut u32) -> i32;
    fn memtype_free(arg0: u64, arg1: u64) -> i32;
    fn pgprot2cachemode(arg0: pgprot) -> u32;
    fn vm_unmap_aliases();
    fn __builtin_unreachable();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
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

fn __kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn kcsan_nestable_atomic_begin() {
}

fn kcsan_nestable_atomic_end() {
}

fn kcsan_atomic_next(n: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


fn kcsan_set_access_mask(mask: u64) {
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn __kcsan_enable_current() {
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


fn __kcsan_disable_current() {
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
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


fn cc_platform_has(attr: u32) -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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
union __SlateRecord_ptrace_h_144 {
    cs: u16,
    csx: u64,
    fred_cs: fred_cs,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_156 {
    ss: u16,
    ssx: u64,
    fred_ss: fred_ss,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
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
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits101U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits102U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pt_regs {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    bp: u64,
    bx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    ax: u64,
    cx: u64,
    dx: u64,
    si: u64,
    di: u64,
    orig_ax: u64,
    ip: u64,
    __slate_anon_17: __SlateRecord_ptrace_h_144,
    flags: u64,
    sp: u64,
    __slate_anon_20: __SlateRecord_ptrace_h_156,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_0() };
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
__verify_pcpu_ptr__3!();
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

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits106U0,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits107U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct leaf_parse_info {
    nr_entries: u32,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_leaves {
    leaf_0x0_0: [leaf_0x0_0; 1],
    leaf_0x0_0_info: leaf_parse_info,
    leaf_0x1_0: [leaf_0x1_0; 1],
    leaf_0x1_0_info: leaf_parse_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_table {
    leaves: cpuid_leaves,
}


#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__2!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__2!().wrapping_sub(unsafe { page_offset_base }) });
x
}

fn __phys_addr_symbol(x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__2!());
// only check upper bounds since lower bounds will trigger carry
VIRTUAL_BUG_ON!();
y.wrapping_add(unsafe { phys_base })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn pfn_to_kaddr(pfn: u64) -> *mut () {
__va!(pfn << 12i32)
}

#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn arch_set_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock orb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) ((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn arch_clear_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock andb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) !((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
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
#[inline(always)]
fn fls(mut x: u32) -> i32 {
let mut r: i32 = 0;
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(x) } {
return if x != 0 { 32i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsrl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[inline(always)]
fn fls64(mut x: u64) -> i32 {
let mut bitpos: i32 = 1i32.wrapping_neg();
if unsafe { __slate_intrinsic_f217c4720f04f6db(x) } {
return if x != 0 { 64i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before.
unsafe {
let __t0: u32;
core::arch::asm!("bsrq {1:r},{0:r}", inlateout(reg) bitpos as u32 => __t0, in(reg) x as u64, options(pure, nomem, nostack, att_syntax));
bitpos = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
bitpos.wrapping_add(1)
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
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 3; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
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
core::arch::asm!("4: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 4b - ., 4; .popsection\n# {0}", const 4 as i32, options(nostack, att_syntax));
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

#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_111 {
    hw_cpu_type: u32,
    __slate_anon_1: __SlateRecord_processor_h_113,
    __slate_anon_2: __SlateRecord_processor_h_119,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits159U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits160U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_136 {
    __slate_anon_0: __SlateRecord_processor_h_142,
    x86_vfm: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_142 {
    x86_model: u8,
    x86: u8,
    x86_vendor: u8,
    x86_reserved: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_154 {
    intel_platform_id: u8,
    amd_unused: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_177 {
    x86_capability: [u32; 24],
    x86_capability_alignment: u64,
}


#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

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
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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


// No need to mask any bits for !PAE
fn native_make_pgd(val: u64) -> __SlateRecord_pgtable_types_h_295 {
let mut __slate_compound_0: __SlateRecord_pgtable_types_h_295 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_types_h_295 { pgd: val & !0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn native_pgd_val(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
pgd.pgd & !0
}

fn pgd_flags(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
native_pgd_val(pgd) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_342 {
    p4d: u64,
}


fn native_make_p4d(val: u64) -> __SlateRecord_pgtable_types_h_342 {
let mut __slate_compound_0: __SlateRecord_pgtable_types_h_342 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_types_h_342 { p4d: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn native_p4d_val(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
p4d.p4d
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


fn native_make_pud(val: u64) -> __SlateRecord_pgtable_types_h_368 {
let mut __slate_compound_0: __SlateRecord_pgtable_types_h_368 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_types_h_368 { pud: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


fn native_pud_val(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
pud.pud
}

fn native_make_pmd(val: u64) -> __SlateRecord_pgtable_64_types_h_22 {
let mut __slate_compound_0: __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_64_types_h_22 { pmd: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn native_pmd_val(pmd: __SlateRecord_pgtable_64_types_h_22) -> u64 {
pmd.pmd
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


fn p4d_pfn_mask(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
// No 512 GiB huge pages yet
((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64
}

fn p4d_flags_mask(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
!p4d_pfn_mask(p4d)
}

fn p4d_flags(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
native_p4d_val(p4d) & p4d_flags_mask(p4d)
}

fn pud_pfn_mask(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
if native_pud_val(pud) & _AT!(u64, 1i32) << 7i32 != 0 {
return ((!(_AC!() << 30i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
} else {
return ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
}
}

fn pud_flags_mask(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
!pud_pfn_mask(pud)
}

fn pud_flags(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
native_pud_val(pud) & pud_flags_mask(pud)
}

fn pmd_pfn_mask(pmd: __SlateRecord_pgtable_64_types_h_22) -> u64 {
if native_pmd_val(pmd) & _AT!(u64, 1i32) << 7i32 != 0 {
return ((!(_AC!() << 21i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
} else {
return ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
}
}

fn pmd_flags_mask(pmd: __SlateRecord_pgtable_64_types_h_22) -> u64 {
!pmd_pfn_mask(pmd)
}

fn pmd_flags(pmd: __SlateRecord_pgtable_64_types_h_22) -> u64 {
native_pmd_val(pmd) & pmd_flags_mask(pmd)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


fn native_make_pte(val: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __slate_compound_0: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_64_types_h_21 { pte: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn native_pte_val(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
pte.pte
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


fn pte_flags(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
native_pte_val(pte) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)
}

fn protval_4k_2_large(val: u64) -> u64 {
val & !(_AT!(u64, 1i32) << 7i32 | _AT!(u64, 1i32) << 12i32) | (val & _AT!(u64, 1i32) << 7i32) << 5i32
}

fn pgprot_4k_2_large(pgprot: pgprot) -> pgprot {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = pgprot { pgprot: protval_4k_2_large(pgprot_val!(pgprot)) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn protval_large_2_4k(val: u64) -> u64 {
val & !(_AT!(u64, 1i32) << 7i32 | _AT!(u64, 1i32) << 12i32) | (val & _AT!(u64, 1i32) << 12i32) >> 5i32
}

fn pgprot_large_2_4k(pgprot: pgprot) -> pgprot {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = pgprot { pgprot: protval_large_2_4k(pgprot_val!(pgprot)) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits138U0,
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


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
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
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_start_locking, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_acquire_returned, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_released, const 0 as u8, label {
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits145U0,
}


#[inline(always)]
fn native_save_fl() -> u64 {
let mut flags: u64 = 0;
// "=rm" is safe here, because "pop" adjusts the stack before
// it evaluates its effective address -- this is part of the
// documented behavior of the "pop" instruction.
unsafe {
let __t0: u64;
core::arch::asm!("# __raw_save_flags\n\tpushf ; pop {0:r}", lateout(reg) __t0, options(nostack, att_syntax));
flags = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
// no input
flags
}

#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
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


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
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

#[inline(always)]
fn list_add(new: *mut list_head, head: *mut list_head) {
__list_add(new, head, unsafe { (*head).next });
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
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

fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[inline(always)]
fn num_online_cpus() -> u32 {
raw_atomic_read((unsafe { core::ptr::addr_of_mut!(__num_online_cpus) }) as *const __SlateRecord_types_h_190) as u32
}

#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn wbinvd() {
unsafe {
core::arch::asm!("wbinvd", options(nostack, att_syntax));
}
}

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


// Instruction encoding provided for binutils backwards compatibility.
fn clflushopt(mut __p: *mut ()) {
unsafe {
core::arch::asm!("# ALT: oldinstr\n771:\n\tds clflush ({0:r})\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 9*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tclflushopt ({0:r})\n775:\n.popsection\n\n# {1}", in(reg) __p as *mut u8, const 0 as i32, options(nostack, att_syntax));
}
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_topology {
    apicid: u32,
    initial_apicid: u32,
    pkg_id: u32,
    die_id: u32,
    cu_id: u32,
    core_id: u32,
    logical_pkg_id: u32,
    logical_die_id: u32,
    logical_core_id: u32,
    amd_node_id: u32,
    llc_id: u32,
    l2c_id: u32,
    __slate_anon_12: __SlateRecord_processor_h_111,
    cpu_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_x86 {
    __slate_anon_0: __SlateRecord_processor_h_136,
    x86_stepping: u8,
    __slate_anon_2: __SlateRecord_processor_h_154,
    x86_tlbsize: i32,
    vmx_capability: [u32; 5],
    x86_virt_bits: u8,
    x86_phys_bits: u8,
    extended_cpuid_level: u32,
    cpuid_level: i32,
    __slate_anon_9: __SlateRecord_processor_h_177,
    x86_vendor_id: [u8; 16],
    x86_model_id: [u8; 64],
    topo: cpuinfo_topology,
    cpuid: cpuid_table,
    x86_cache_size: u32,
    x86_cache_alignment: i32,
    x86_cache_max_rmid: i32,
    x86_cache_occ_scale: i32,
    x86_cache_mbm_width_offset: i32,
    x86_power: i32,
    loops_per_jiffy: u64,
    ppin: u64,
    x86_clflush_size: u16,
    booted_cores: u16,
    cpu_index: u16,
    smt_active: bool,
    microcode: u32,
    x86_cache_bits: u8,
    __slate_bits_0: __slate_bits::__SlateBits105U0,
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


#[inline(always)]
fn __static_cpu_has__const4(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+77(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 235 as u16, const 8 as i32, sym boot_cpu_data, label {
__slate_asm_target = 1usize;
}, label {
__slate_asm_target = 2usize;
}, options(nostack, att_syntax));
if __slate_asm_target != 1usize {
if __slate_asm_target == 2usize {
return false;
}
}
return true;
}
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
fn __static_cpu_has__const5(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+114(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 528 as u16, const 1 as i32, sym boot_cpu_data, label {
__slate_asm_target = 1usize;
}, label {
__slate_asm_target = 2usize;
}, options(nostack, att_syntax));
if __slate_asm_target != 1usize {
if __slate_asm_target == 2usize {
return false;
}
}
return true;
}
}

#[inline(always)]
fn __static_cpu_has__const6(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+67(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 159 as u16, const 128 as i32, sym boot_cpu_data, label {
__slate_asm_target = 1usize;
}, label {
__slate_asm_target = 2usize;
}, options(nostack, att_syntax));
if __slate_asm_target != 1usize {
if __slate_asm_target == 2usize {
return false;
}
}
return true;
}
}

#[inline(always)]
fn __static_cpu_has__const8(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+50(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 19 as u16, const 8 as i32, sym boot_cpu_data, label {
__slate_asm_target = 1usize;
}, label {
__slate_asm_target = 2usize;
}, options(nostack, att_syntax));
if __slate_asm_target != 1usize {
if __slate_asm_target == 2usize {
return false;
}
}
return true;
}
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
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


fn on_each_cpu(func: Option<unsafe extern "C" fn(*mut ())>, info: *mut (), wait: i32) {
unsafe { on_each_cpu_cond_mask(None, func, info, wait != 0, core::ptr::addr_of_mut!(__cpu_online_mask) as *const cpumask) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct user_namespace {
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
struct mem_cgroup {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits252U0,
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
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
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


#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct __SlateRecord_spinlock_h_610 {
    lock: *mut spinlock,
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
union __SlateRecord_mm_types_h_579 {
    pt_rcu_head: callback_head,
    pt_list: list_head,
    __slate_anon_2: __SlateRecord_mm_types_h_582,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_582 {
    _pt_pad_1: u64,
    pmd_huge_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_589 {
    pt_index: u64,
    pt_mm: *mut mm_struct,
    pt_frag_refcount: __SlateRecord_types_h_190,
    pt_share_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_598 {
    _pt_pad_2: u64,
    ptl: spinlock,
}


#[inline(always)]
fn class_spinlock_destructor(_T: *mut __SlateRecord_spinlock_h_610) {
spin_unlock(unsafe { (*_T).lock });
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
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
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
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
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
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
struct memory_tier {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ghcb {
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
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
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
struct mnt_idmap {
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
    __slate_bits_0: __slate_bits::__SlateBits318U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct kmap_ctrl {
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
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
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
    __slate_bits_0: __slate_bits::__SlateBits70U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits70U1,
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
    __slate_bits_2: __slate_bits::__SlateBits70U2,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


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
fn do_raw_write_seqcount_begin(__v0: *mut seqcount) {
kcsan_nestable_atomic_begin();
let __v1: u32 = unsafe { (*__v0).sequence };
unsafe {
(*__v0).sequence = __v1.wrapping_add(1);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}

fn do_raw_write_seqcount_end(__v2: *mut seqcount) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let __v3: u32 = unsafe { (*__v2).sequence };
unsafe {
(*__v2).sequence = __v3.wrapping_add(1);
}
kcsan_nestable_atomic_end();
}

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
#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn rwsem_assert_held_write_nolockdep(sem: *const rw_semaphore) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((atomic_long_read(unsafe { core::ptr::addr_of!((*sem).count) }) as u64) & 1u64 == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("156: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 156b - ., 3; .popsection\n# {0}", const 156 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rwsem_assert_held_write_nolockdep_1, sym __slate_asm_string_rwsem_assert_held_write_nolockdep_2, const 87 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("157: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 157b - ., 4; .popsection\n# {0}", const 157 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn rwsem_assert_held_write(sem: *const rw_semaphore) {
if IS_ENABLED!() != 0 {
lockdep_assert_held_write!(sem);
} else {
rwsem_assert_held_write_nolockdep(sem);
}
}

// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
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
struct uprobes_state {
    xol_area: *mut xol_area,
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
struct fs_context {
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
struct ptdesc {
    pt_flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_579,
    __page_mapping: u64,
    __slate_anon_3: __SlateRecord_mm_types_h_589,
    __slate_anon_4: __SlateRecord_mm_types_h_598,
    __page_type: u32,
    __page_refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


fn ptdesc_pmd_pts_init(ptdesc: *mut ptdesc) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*ptdesc).__slate_anon_3.pt_share_count) }, 0i32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits448U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
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
// page_folio - Converts from page to folio.
// @p: The page.
//
// Every page is part of a folio.  This function cannot be called on a
// NULL pointer.
//
// Context: No reference, nor lock is required on @page.  If the caller
// does not hold a reference, this call may race with a folio split, so
// it should re-check the folio still contains this page after gaining
// a reference on the folio.
// Return: The folio which contains this page.
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
fn const_folio_flags(folio: *const folio, n: u32) -> *const u64 {
let page: *const page = unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of!((*page.wrapping_offset(n as isize)).flags.f) }
}

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
fn PageHighMem(page: *const page) -> i32 {
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


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
#[inline(always)]
fn folio_test_head(folio: *const folio) -> bool {
test_bit__6!(6i32, const_folio_flags(folio, 0u32)) != 0
}

fn folio_test_large(folio: *const folio) -> bool {
folio_test_head(folio)
}

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


#[inline(always)]
fn folio_test_pgtable(folio: *const folio) -> bool {
data_race!((unsafe { (*folio).__slate_anon_0.page.__slate_anon_2.page_type }) >> 24i32) == 242
}

#[inline(always)]
fn __folio_set_pgtable(folio: *mut folio) {
if folio_test_pgtable(folio as *const folio) {
return;
}
VM_BUG_ON_FOLIO!();
unsafe {
(*folio).__slate_anon_0.page.__slate_anon_2.page_type = 4060086272u32;
}
}

#[inline(always)]
fn __folio_clear_pgtable(folio: *mut folio) {
if (unsafe { (*folio).__slate_anon_0.page.__slate_anon_2.page_type }) == !0 {
return;
}
VM_BUG_ON_FOLIO!();
unsafe {
(*folio).__slate_anon_0.page.__slate_anon_2.page_type = !0;
}
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
fn PageTable(page: *const page) -> i32 {
(data_race!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 242) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zswap_lruvec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct free_area {
    free_list: [list_head; 4],
    nr_free: u64,
}


// We do arithmetic on the LRU lists in various places in the code,
// so it is important to keep the active lists LRU_ACTIVE higher in
// the array than the corresponding inactive lists, and to keep
// the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//
// This has to be kept in sync with the statistics in zone_stat_item
// above and the descriptions in vmstat_text in mm/vmstat.c
#[repr(C)]
#[derive(Clone, Copy)]
struct lru_cost {
    count: u64,
    last_rotated: u64,
    last_io: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lruvec {
    lists: [list_head; 5],
    lru_lock: spinlock,
    cost: [lru_cost; 2],
    cost_lock: spinlock,
    nonresident_age: __SlateRecord_types_h_197,
    refaults: [u64; 2],
    flags: u64,
    zswap_lruvec_state: zswap_lruvec_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct per_cpu_pages {
    lock: spinlock,
    count: i32,
    high: i32,
    high_min: i32,
    high_max: i32,
    batch: i32,
    flags: u8,
    alloc_factor: u8,
    expire: u8,
    free_count: i16,
    lists: [list_head; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_zonestat {
    vm_stat_diff: [i8; 10],
    stat_threshold: i8,
    vm_numa_event: [u64; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_nodestat {
    stat_threshold: i8,
    vm_node_stat_diff: [i8; 66],
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


#[repr(C)]
#[derive(Clone, Copy)]
struct zone {
    _watermark: [u64; 4],
    watermark_boost: u64,
    nr_reserved_highatomic: u64,
    nr_free_highatomic: u64,
    lowmem_reserve: [i64; 4],
    node: i32,
    zone_pgdat: *mut pglist_data,
    per_cpu_pageset: *mut per_cpu_pages,
    per_cpu_zonestats: *mut per_cpu_zonestat,
    pageset_high_min: i32,
    pageset_high_max: i32,
    pageset_batch: i32,
    zone_start_pfn: u64,
    managed_pages: __SlateRecord_types_h_197,
    spanned_pages: u64,
    present_pages: u64,
    name: *const u8,
    initialized: i32,
    _pad1_: cacheline_padding,
    free_area: [free_area; 11],
    flags: u64,
    lock: spinlock,
    trylock_free_pages: llist_head,
    _pad2_: cacheline_padding,
    percpu_drift_mark: u64,
    compact_cached_free_pfn: u64,
    compact_cached_migrate_pfn: [u64; 2],
    compact_init_migrate_pfn: u64,
    compact_init_free_pfn: u64,
    compact_considered: u32,
    compact_defer_shift: u32,
    compact_order_failed: i32,
    compact_blockskip_flush: bool,
    contiguous: bool,
    _pad3_: cacheline_padding,
    vm_stat: [__SlateRecord_types_h_197; 10],
    vm_numa_event: [__SlateRecord_types_h_197; 6],
    vmemmap_tails: [*mut page; 16],
}


// Maximum number of zones on a zonelist
#[repr(C)]
#[derive(Clone, Copy)]
struct zoneref {
    zone: *mut zone,
    zone_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zonelist {
    _zonerefs: [zoneref; 257],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pglist_data {
    node_zones: [zone; 4],
    node_zonelists: [zonelist; 2],
    nr_zones: i32,
    node_start_pfn: u64,
    node_present_pages: u64,
    node_spanned_pages: u64,
    node_id: i32,
    kswapd_wait: wait_queue_head,
    pfmemalloc_wait: wait_queue_head,
    reclaim_wait: [wait_queue_head; 4],
    nr_writeback_throttled: __SlateRecord_types_h_190,
    nr_reclaim_start: u64,
    kswapd: *mut task_struct,
    kswapd_order: i32,
    kswapd_highest_zoneidx: u32,
    kswapd_failures: __SlateRecord_types_h_190,
    kcompactd_max_order: i32,
    kcompactd_highest_zoneidx: u32,
    kcompactd_wait: wait_queue_head,
    kcompactd: *mut task_struct,
    proactive_compact_trigger: bool,
    totalreserve_pages: u64,
    min_unmapped_pages: u64,
    min_slab_pages: u64,
    _pad1_: cacheline_padding,
    __lruvec: lruvec,
    flags: u64,
    _pad2_: cacheline_padding,
    per_cpu_nodestats: *mut per_cpu_nodestat,
    vm_stat: [__SlateRecord_types_h_197; 66],
    memtier: *mut memory_tier,
}


// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
#[repr(C)]
#[derive(Clone, Copy)]
struct x86_guest {
    enc_status_change_prepare: Option<unsafe extern "C" fn(u64, i32, bool) -> i32>,
    enc_status_change_finish: Option<unsafe extern "C" fn(u64, i32, bool) -> i32>,
    enc_tlb_flush_required: Option<unsafe extern "C" fn(bool) -> bool>,
    enc_cache_flush_required: Option<unsafe extern "C" fn() -> bool>,
    enc_kexec_begin: Option<unsafe extern "C" fn()>,
    enc_kexec_finish: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_legacy_devices {
    pnpbios: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_legacy_features {
    i8042: u32,
    rtc: i32,
    warm_reset: i32,
    no_vga: i32,
    reserve_bios_regions: i32,
    devices: x86_legacy_devices,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_hyper_runtime {
    pin_vcpu: Option<unsafe extern "C" fn(i32)>,
    sev_es_hcall_prepare: Option<unsafe extern "C" fn(*mut ghcb, *mut pt_regs)>,
    sev_es_hcall_finish: Option<unsafe extern "C" fn(*mut ghcb, *mut pt_regs) -> bool>,
    is_private_mmio: Option<unsafe extern "C" fn(u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_platform_ops {
    calibrate_cpu: Option<unsafe extern "C" fn() -> u64>,
    calibrate_tsc: Option<unsafe extern "C" fn() -> u64>,
    get_wallclock: Option<unsafe extern "C" fn(*mut timespec64)>,
    set_wallclock: Option<unsafe extern "C" fn(*const timespec64) -> i32>,
    iommu_shutdown: Option<unsafe extern "C" fn()>,
    is_untracked_pat_range: Option<unsafe extern "C" fn(u64, u64) -> bool>,
    nmi_init: Option<unsafe extern "C" fn()>,
    get_nmi_reason: Option<unsafe extern "C" fn() -> u8>,
    save_sched_clock_state: Option<unsafe extern "C" fn()>,
    restore_sched_clock_state: Option<unsafe extern "C" fn()>,
    apic_post_init: Option<unsafe extern "C" fn()>,
    legacy: x86_legacy_features,
    set_legacy_features: Option<unsafe extern "C" fn()>,
    realmode_reserve: Option<unsafe extern "C" fn()>,
    realmode_init: Option<unsafe extern "C" fn()>,
    hyper: x86_hyper_runtime,
    guest: x86_guest,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
// xas_for_each() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
//
// The loop body will be executed for each entry present in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
fn __mmap_lock_trace_start_locking(mm: *mut mm_struct, write: bool) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_start_locking.key) }) != 0 {
unsafe { __mmap_lock_do_trace_start_locking(mm, write) };
}
}

fn __mmap_lock_trace_acquire_returned(mm: *mut mm_struct, write: bool, success: bool) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_acquire_returned.key) }) != 0 {
unsafe { __mmap_lock_do_trace_acquire_returned(mm, write, success) };
}
}

fn __mmap_lock_trace_released(mm: *mut mm_struct, write: bool) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_released.key) }) != 0 {
unsafe { __mmap_lock_do_trace_released(mm, write) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


fn mmap_assert_write_locked(mm: *const mm_struct) {
rwsem_assert_held_write(unsafe { core::ptr::addr_of!((*mm).__slate_anon_0.mmap_lock) });
}

// VMA locks do not behave like most ordinary locks found in the kernel, so we
// cannot quite have full lockdep tracking in the way we would ideally prefer.
//
// Read locks act as shared locks which exclude an exclusive lock being
// taken. We therefore mark these accordingly on read lock acquire/release.
//
// Write locks are acquired exclusively per-VMA, but released in a shared
// fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
// that write lock is released.
//
// We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
// is the fact that, of course, we do lockdep-track the mmap lock rwsem which
// must be held when taking a VMA write lock.
//
// We do, however, want to indicate that during either acquisition of a VMA
// write lock or detachment of a VMA that we require the lock held be exclusive,
// so we utilise lockdep to do so.
fn mm_lock_seqcount_begin(mm: *mut mm_struct) {
do_raw_write_seqcount_begin(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_lock_seq) });
}

fn mm_lock_seqcount_end(mm: *mut mm_struct) {
ASSERT_EXCLUSIVE_WRITER!((*mm).__slate_anon_0.mm_lock_seq);
do_raw_write_seqcount_end(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_lock_seq) });
}

fn mmap_write_lock(mm: *mut mm_struct) {
__mmap_lock_trace_start_locking(mm, true);
unsafe { down_write(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
mm_lock_seqcount_begin(mm);
__mmap_lock_trace_acquire_returned(mm, true, true);
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


fn vma_end_write_all(mm: *mut mm_struct) {
mmap_assert_write_locked(mm as *const mm_struct);
mm_lock_seqcount_end(mm);
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits318U0 {
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
    pub struct __SlateBits70U0 {
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
    pub struct __SlateBits70U1 {
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
    pub struct __SlateBits321U0 {
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
    pub struct __SlateBits70U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits138U0 {
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
    pub struct __SlateBits618U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits684U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits625U0 {
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
    pub struct __SlateBits448U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits159U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits160U0 {
        #[bits(16)]
        pub amd_num_processors: u32,
        #[bits(8)]
        pub amd_power_eff_ranking: u32,
        #[bits(4)]
        pub amd_native_model_id: u32,
        #[bits(4)]
        pub amd_type: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits106U0 {
        #[bits(32)]
        pub max_std_leaf: u32,
        #[bits(32)]
        pub cpu_vendorid_0: u32,
        #[bits(32)]
        pub cpu_vendorid_2: u32,
        #[bits(32)]
        pub cpu_vendorid_1: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits107U0 {
        #[bits(4)]
        pub stepping: u32,
        #[bits(4)]
        pub base_model: u32,
        #[bits(4)]
        pub base_family_id: u32,
        #[bits(2)]
        pub cpu_type: u32,
        #[bits(2, access = na)]
        pub __slate_pad_4: u8,
        #[bits(4)]
        pub ext_model: u32,
        #[bits(8)]
        pub ext_family: u32,
        #[bits(4, access = na)]
        pub __slate_pad_7: u8,
        #[bits(8)]
        pub brand_id: u32,
        #[bits(8)]
        pub clflush_size: u32,
        #[bits(8)]
        pub n_logical_cpu: u32,
        #[bits(8)]
        pub local_apic_id: u32,
        #[bits(1)]
        pub sse3: u32,
        #[bits(1)]
        pub pclmulqdq: u32,
        #[bits(1)]
        pub dtes64: u32,
        #[bits(1)]
        pub monitor: u32,
        #[bits(1)]
        pub dscpl: u32,
        #[bits(1)]
        pub vmx: u32,
        #[bits(1)]
        pub smx: u32,
        #[bits(1)]
        pub est: u32,
        #[bits(1)]
        pub tm2: u32,
        #[bits(1)]
        pub ssse3: u32,
        #[bits(1)]
        pub cntxt_id: u32,
        #[bits(1)]
        pub sdbg: u32,
        #[bits(1)]
        pub fma: u32,
        #[bits(1)]
        pub cx16: u32,
        #[bits(1)]
        pub xtpr_update: u32,
        #[bits(1)]
        pub pdcm: u32,
        #[bits(1, access = na)]
        pub __slate_pad_28: u8,
        #[bits(1)]
        pub pcid: u32,
        #[bits(1)]
        pub dca: u32,
        #[bits(1)]
        pub sse4_1: u32,
        #[bits(1)]
        pub sse4_2: u32,
        #[bits(1)]
        pub x2apic: u32,
        #[bits(1)]
        pub movbe: u32,
        #[bits(1)]
        pub popcnt: u32,
        #[bits(1)]
        pub tsc_deadline_timer: u32,
        #[bits(1)]
        pub aes: u32,
        #[bits(1)]
        pub xsave: u32,
        #[bits(1)]
        pub osxsave: u32,
        #[bits(1)]
        pub avx: u32,
        #[bits(1)]
        pub f16c: u32,
        #[bits(1)]
        pub rdrand: u32,
        #[bits(1)]
        pub guest_status: u32,
        #[bits(1)]
        pub fpu: u32,
        #[bits(1)]
        pub vme: u32,
        #[bits(1)]
        pub de: u32,
        #[bits(1)]
        pub pse: u32,
        #[bits(1)]
        pub tsc: u32,
        #[bits(1)]
        pub msr: u32,
        #[bits(1)]
        pub pae: u32,
        #[bits(1)]
        pub mce: u32,
        #[bits(1)]
        pub cx8: u32,
        #[bits(1)]
        pub apic: u32,
        #[bits(1, access = na)]
        pub __slate_pad_54: u8,
        #[bits(1)]
        pub sep: u32,
        #[bits(1)]
        pub mtrr: u32,
        #[bits(1)]
        pub pge: u32,
        #[bits(1)]
        pub mca: u32,
        #[bits(1)]
        pub cmov: u32,
        #[bits(1)]
        pub pat: u32,
        #[bits(1)]
        pub pse36: u32,
        #[bits(1)]
        pub psn: u32,
        #[bits(1)]
        pub clflush: u32,
        #[bits(1, access = na)]
        pub __slate_pad_64: u8,
        #[bits(1)]
        pub ds: u32,
        #[bits(1)]
        pub acpi: u32,
        #[bits(1)]
        pub mmx: u32,
        #[bits(1)]
        pub fxsr: u32,
        #[bits(1)]
        pub sse: u32,
        #[bits(1)]
        pub sse2: u32,
        #[bits(1)]
        pub selfsnoop: u32,
        #[bits(1)]
        pub htt: u32,
        #[bits(1)]
        pub tm: u32,
        #[bits(1)]
        pub ia64: u32,
        #[bits(1)]
        pub pbe: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits105U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits101U0 {
        #[bits(16)]
        pub cs: u64,
        #[bits(2)]
        pub sl: u64,
        #[bits(1)]
        pub wfe: u64,
        #[bits(45, access = na)]
        pub __slate_pad_3: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits102U0 {
        #[bits(16)]
        pub ss: u64,
        #[bits(1)]
        pub sti: u64,
        #[bits(1)]
        pub swevent: u64,
        #[bits(1)]
        pub nmi: u64,
        #[bits(13, access = na)]
        pub __slate_pad_4: u16,
        #[bits(8)]
        pub vector: u64,
        #[bits(8, access = na)]
        pub __slate_pad_6: u8,
        #[bits(4)]
        pub r#type: u64,
        #[bits(4, access = na)]
        pub __slate_pad_8: u8,
        #[bits(1)]
        pub enclave: u64,
        #[bits(1)]
        pub l: u64,
        #[bits(1)]
        pub nested: u64,
        #[bits(1, access = na)]
        pub __slate_pad_12: u8,
        #[bits(4)]
        pub insnlen: u64,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits252U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits743U0 {
        #[bits(1)]
        pub force_split: u32,
        #[bits(1)]
        pub force_static_prot: u32,
        #[bits(1)]
        pub force_flush_all: u32,
        #[bits(1)]
        pub init_mm_read_locked: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
}

fn mmap_write_unlock(mm: *mut mm_struct) {
__mmap_lock_trace_released(mm, true);
vma_end_write_all(mm);
unsafe { up_write(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
}

fn mmap_read_lock(mm: *mut mm_struct) {
__mmap_lock_trace_start_locking(mm, false);
unsafe { down_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
__mmap_lock_trace_acquire_returned(mm, false, true);
}

fn mmap_read_unlock(mm: *mut mm_struct) {
__mmap_lock_trace_released(mm, false);
unsafe { up_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
}

#[inline(always)]
fn class_mmap_read_lock_destructor(p: *mut *mut mm_struct) {
let _T: *mut mm_struct = unsafe { *p };
mmap_read_unlock(_T);
}

#[inline(always)]
fn class_mmap_read_lock_constructor__const9(_T: *mut mm_struct) -> *mut mm_struct {
let mut __v0: *mut mm_struct;
mmap_read_lock(_T);
__v0 = _T;
let t: *mut mm_struct = __v0;
t
}

static class_mmap_read_lock_is_conditional: bool = 0i32 != 0i32;

#[inline(always)]
fn class_mmap_read_lock_lock_ptr(_T: &*mut mm_struct) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *(_T as *const *mut mm_struct) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_mmap_write_lock_destructor(p: *mut *mut mm_struct) {
let _T: *mut mm_struct = unsafe { *p };
mmap_write_unlock(_T);
}

#[inline(always)]
fn class_mmap_write_lock_constructor__const7(_T: *mut mm_struct) -> *mut mm_struct {
let mut __v0: *mut mm_struct;
mmap_write_lock(_T);
__v0 = _T;
let t: *mut mm_struct = __v0;
t
}

static class_mmap_write_lock_is_conditional: bool = 0i32 != 0i32;

#[inline(always)]
fn class_mmap_write_lock_lock_ptr(_T: &*mut mm_struct) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *(_T as *const *mut mm_struct) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


fn cc_mkenc(val: u64) -> u64 {
val
}

fn cc_mkdec(val: u64) -> u64 {
val
}

fn pmd_set_flags(pmd: __SlateRecord_pgtable_64_types_h_22, set: u64) -> __SlateRecord_pgtable_64_types_h_22 {
let v: u64 = native_pmd_val(pmd);
return native_make_pmd(v | set);
}

fn pud_set_flags(pud: __SlateRecord_pgtable_types_h_368, set: u64) -> __SlateRecord_pgtable_types_h_368 {
let v: u64 = native_pud_val(pud);
return native_make_pud(v | set);
}

fn pte_pfn(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
let mut pfn: u64 = pte_val!(pte);
let __v0: u64 = protnone_mask(pfn);
pfn ^= __v0;
(pfn & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)) >> 12i32
}

fn pmd_pfn(pmd: __SlateRecord_pgtable_64_types_h_22) -> u64 {
let mut pfn: u64 = pmd_val!(pmd);
let __v0: u64 = protnone_mask(pfn);
pfn ^= __v0;
(pfn & pmd_pfn_mask(pmd)) >> 12i32
}

fn pud_pfn(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
let mut pfn: u64 = pud_val!(pud);
let __v0: u64 = protnone_mask(pfn);
pfn ^= __v0;
(pfn & pud_pfn_mask(pud)) >> 12i32
}

fn pmd_leaf(pte: __SlateRecord_pgtable_64_types_h_22) -> bool {
pmd_flags(pte) & _AT!(u64, 1i32) << 7i32 != 0
}

fn pmd_mkhuge(pmd: __SlateRecord_pgtable_64_types_h_22) -> __SlateRecord_pgtable_64_types_h_22 {
return pmd_set_flags(pmd, _AT!(u64, 1i32) << 7i32);
}

fn pud_mkhuge(pud: __SlateRecord_pgtable_types_h_368) -> __SlateRecord_pgtable_types_h_368 {
return pud_set_flags(pud, _AT!(u64, 1i32) << 7i32);
}

fn massage_pgprot(pgprot: pgprot) -> u64 {
let mut protval: u64 = pgprot_val!(pgprot);
if protval & _AT!(u64, 1i32) << 0i32 != 0 {
protval &= unsafe { __supported_pte_mask };
}
protval
}

fn check_pgprot(pgprot: pgprot) -> u64 {
let massaged_val: u64 = massage_pgprot(pgprot);
// mmdebug.h can not be included here because of dependencies
massaged_val
}

fn pfn_pte(page_nr: u64, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
let mut pfn: u64 = page_nr << 12i32;
// This bit combination is used to mark shadow stacks
let mut __v0: i64;
let __ret_warn_on: i32 = (pgprot_val!(pgprot) & (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 1i32) == _AT!(u64, 1i32) << 6i32) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("286: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 286b - ., 3; .popsection\n# {0}", const 286 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pfn_pte_1, sym __slate_asm_string_pfn_pte_2, const 735 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("287: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 287b - ., 4; .popsection\n# {0}", const 287 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
let __v5: u64 = protnone_mask(pgprot_val!(pgprot));
pfn ^= __v5;
let __v8: u64 = pfn;
pfn = __v8 & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
return __pte!(pfn | check_pgprot(pgprot));
}

fn pfn_pmd(page_nr: u64, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_22 {
let mut pfn: u64 = page_nr << 12i32;
let __v0: u64 = protnone_mask(pgprot_val!(pgprot));
pfn ^= __v0;
let __v3: u64 = pfn;
pfn = __v3 & (((!(_AC!() << 21i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
return __pmd!(pfn | check_pgprot(pgprot));
}

fn pfn_pud(page_nr: u64, pgprot: pgprot) -> __SlateRecord_pgtable_types_h_368 {
let mut pfn: u64 = page_nr << 12i32;
let __v0: u64 = protnone_mask(pgprot_val!(pgprot));
pfn ^= __v0;
let __v3: u64 = pfn;
pfn = __v3 & (((!(_AC!() << 30i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
return __pud!(pfn | check_pgprot(pgprot));
}

// mprotect needs to preserve PAT and encryption bits when updating
// vm_page_prot
fn pti_set_user_pgtbl(pgdp: *mut __SlateRecord_pgtable_types_h_295, pgd: __SlateRecord_pgtable_types_h_295) -> __SlateRecord_pgtable_types_h_295 {
if cpu_feature_enabled!(235i32) == 0 {
return pgd;
}
return unsafe { __pti_set_user_pgtbl(pgdp, pgd) };
}

fn native_set_pte(ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21) {
loop {
loop {
if !(__native_word__16!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_290() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__9!(*ptep, pte);
if true {
break;
}
}
}

fn native_set_pte_atomic(ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21) {
native_set_pte(ptep, pte);
}

fn native_set_pmd(pmdp: *mut __SlateRecord_pgtable_64_types_h_22, pmd: __SlateRecord_pgtable_64_types_h_22) {
loop {
loop {
if !(__native_word__12!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_22>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_291() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!(*pmdp, pmd);
if true {
break;
}
}
}

fn native_pmd_clear(pmd: *mut __SlateRecord_pgtable_64_types_h_22) {
native_set_pmd(pmd, native_make_pmd(0u64));
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_clflush_cache_range: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_arch_invalidate_pmem: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_cpu_cache_has_invalidate_memregion: ; .asciz \"GPL\" ; .ascii \"DEVMEM\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_cpu_cache_invalidate_memregion: ; .asciz \"GPL\" ; .ascii \"DEVMEM\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_lookup_address: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_slow_virt_to_phys: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_memory_uc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_memory_wc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_memory_wb: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_mce_nospec: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_clear_mce_nospec: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_memory_encrypted: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_memory_decrypted: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_pages_uc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_pages_array_uc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_pages_array_wc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_pages_wb: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_set_pages_array_wb: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous", sym clflush_cache_range, sym arch_invalidate_pmem, sym cpu_cache_has_invalidate_memregion, sym cpu_cache_invalidate_memregion, sym lookup_address, sym slow_virt_to_phys, sym set_memory_uc, sym set_memory_wc, sym set_memory_wb, sym set_mce_nospec, sym clear_mce_nospec, sym set_memory_encrypted, sym set_memory_decrypted, sym set_pages_uc, sym set_pages_array_uc, sym set_pages_array_wc, sym set_pages_wb, sym set_pages_array_wb, options(att_syntax));

fn native_set_pud(pudp: *mut __SlateRecord_pgtable_types_h_368, pud: __SlateRecord_pgtable_types_h_368) {
loop {
loop {
if !(__native_word__13!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_types_h_368>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_292() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!(*pudp, pud);
if true {
break;
}
}
}

fn native_pud_clear(pud: *mut __SlateRecord_pgtable_types_h_368) {
native_set_pud(pud, native_make_pud(0u64));
}

fn native_set_p4d(p4dp: *mut __SlateRecord_pgtable_types_h_342, p4d: __SlateRecord_pgtable_types_h_342) {
let mut pgd: __SlateRecord_pgtable_types_h_295 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pgd) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
if pgtable_l5_enabled__3!() != 0 || IS_ENABLED__2!() == 0 {
loop {
while false {
}
__WRITE_ONCE__4!(*p4dp, p4d);
if true {
break;
}
}
return;
}
pgd = native_make_pgd(native_p4d_val(p4d));
pgd = pti_set_user_pgtbl(p4dp as *mut __SlateRecord_pgtable_types_h_295, pgd);
loop {
while false {
}
__WRITE_ONCE__4!(*p4dp, native_make_p4d(native_pgd_val(pgd)));
if true {
break;
}
}
}

fn native_set_pgd(pgdp: *mut __SlateRecord_pgtable_types_h_295, pgd: __SlateRecord_pgtable_types_h_295) {
loop {
while false {
}
__WRITE_ONCE__7!(*pgdp, pti_set_user_pgtbl(pgdp, pgd));
if true {
break;
}
}
}

fn __pte_needs_invert(val: u64) -> bool {
val != 0 && val & _AT!(u64, 1i32) << 0i32 == 0
}

fn protnone_mask(val: u64) -> u64 {
if __pte_needs_invert(val) { !0 } else { 0 }
}

fn pte_none(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte.pte & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn pte_present(a: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(a) & (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 8i32)) as i32
}

fn pmd_present(pmd: __SlateRecord_pgtable_64_types_h_22) -> i32 {
// Checking for _PAGE_PSE is needed too because
// split_huge_page will temporarily clear the present bit (but
// the _PAGE_PSE flag will remain set at all times while the
// _PAGE_PRESENT bit is clear).
(pmd_flags(pmd) & (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 8i32 | _AT!(u64, 1i32) << 7i32)) as i32
}

fn pmd_none(pmd: __SlateRecord_pgtable_64_types_h_22) -> i32 {
// Only check low word on 32-bit platforms, since it might be
// out of sync with upper half.
let val: u64 = native_pmd_val(pmd);
(val & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn pmd_page_vaddr(pmd: __SlateRecord_pgtable_64_types_h_22) -> u64 {
__va!(pmd_val!(pmd) & pmd_pfn_mask(pmd)) as u64
}

// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
fn pud_none(pud: __SlateRecord_pgtable_types_h_368) -> i32 {
(native_pud_val(pud) & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn pud_present(pud: __SlateRecord_pgtable_types_h_368) -> i32 {
(pud_flags(pud) & _AT!(u64, 1i32) << 0i32) as i32
}

fn pud_pgtable(pud: __SlateRecord_pgtable_types_h_368) -> *mut __SlateRecord_pgtable_64_types_h_22 {
__va!(pud_val!(pud) & pud_pfn_mask(pud)) as *mut __SlateRecord_pgtable_64_types_h_22
}

// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
fn pud_leaf(pud: __SlateRecord_pgtable_types_h_368) -> bool {
pud_val!(pud) & _AT!(u64, 1i32) << 7i32 != 0
}

fn p4d_none(p4d: __SlateRecord_pgtable_types_h_342) -> i32 {
(native_p4d_val(p4d) & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn p4d_present(p4d: __SlateRecord_pgtable_types_h_342) -> i32 {
(p4d_flags(p4d) & _AT!(u64, 1i32) << 0i32) as i32
}

fn p4d_pgtable(p4d: __SlateRecord_pgtable_types_h_342) -> *mut __SlateRecord_pgtable_types_h_368 {
__va!(p4d_val!(p4d) & p4d_pfn_mask(p4d)) as *mut __SlateRecord_pgtable_types_h_368
}

fn p4d_index(address: u64) -> u64 {
address >> 39i32 & (unsafe { ptrs_per_p4d }.wrapping_sub(1) as u64)
}

fn pgd_page_vaddr(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
__va!(pgd_val!(pgd) & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)) as u64
}

fn p4d_offset(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_342 {
if pgtable_l5_enabled__3!() == 0 {
return pgd as *mut __SlateRecord_pgtable_types_h_342;
}
(pgd_page_vaddr(unsafe { *pgd }) as *mut __SlateRecord_pgtable_types_h_342).wrapping_offset(p4d_index(address) as isize)
}

fn pgd_none(pgd: __SlateRecord_pgtable_types_h_295) -> i32 {
if pgtable_l5_enabled__3!() == 0 {
return 0;
}
// There is no need to do a workaround for the KNL stray
// A/D bit erratum here.  PGDs only point to page tables
// except on 32-bit non-PAE which is not supported on
// KNL.
(native_pgd_val(pgd) == 0) as i32
}

// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
fn page_level_shift(level: u32) -> i32 {
(12i32.wrapping_sub(ilog2__2!(512i32)) as u32).wrapping_add((level as u32).wrapping_mul(ilog2__2!(512i32) as u32)) as i32
}

fn page_level_size(level: u32) -> u64 {
1u64 << page_level_shift(level)
}

fn page_level_mask(level: u32) -> u64 {
!page_level_size(level).wrapping_sub(1)
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
struct rhash_head {
    next: *mut rhash_head,
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
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

fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
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
    __slate_bits_0: __slate_bits::__SlateBits321U0,
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


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_359() };
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
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
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
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_384() };
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
    __slate_bits_0: __slate_bits::__SlateBits618U0,
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
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


fn pte_index(address: u64) -> u64 {
address >> 12i32 & 511u64
}

fn pmd_index(address: u64) -> u64 {
address >> 21i32 & 511u64
}

fn pud_index(address: u64) -> u64 {
address >> 30i32 & 511u64
}

fn pte_offset_kernel(pmd: *mut __SlateRecord_pgtable_64_types_h_22, address: u64) -> *mut __SlateRecord_pgtable_64_types_h_21 {
(pmd_page_vaddr(unsafe { *pmd }) as *mut __SlateRecord_pgtable_64_types_h_21).wrapping_offset(pte_index(address) as isize)
}

fn pmd_offset(pud: *mut __SlateRecord_pgtable_types_h_368, address: u64) -> *mut __SlateRecord_pgtable_64_types_h_22 {
pud_pgtable(unsafe { *pud }).wrapping_offset(pmd_index(address) as isize)
}

fn pud_offset(p4d: *mut __SlateRecord_pgtable_types_h_342, address: u64) -> *mut __SlateRecord_pgtable_types_h_368 {
p4d_pgtable(unsafe { *p4d }).wrapping_offset(pud_index(address) as isize)
}

fn pgd_offset_pgd(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_295 {
pgd.wrapping_offset(pgd_index!(address) as isize)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cacheinfo {
    id: u32,
    r#type: u32,
    level: u32,
    coherency_line_size: u32,
    number_of_sets: u32,
    ways_of_associativity: u32,
    physical_line_partition: u32,
    size: u32,
    shared_cpu_map: cpumask,
    attributes: u32,
    fw_token: *mut (),
    disable_sysfs: bool,
    r#priv: *mut (),
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
fn folio_large_order(folio: *const folio) -> u32 {
((unsafe { (*folio).__slate_anon_1.__slate_anon_0._flags_1 }) & 255) as u32
}

fn folio_large_nr_pages(folio: *const folio) -> u64 {
(1i64 << folio_large_order(folio)) as u64
}

fn compound_order(page: *const page) -> u32 {
let folio: *const folio = page as *const folio;
if test_bit__4!(6i32, unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags.f) }) == 0 {
return 0;
}
folio_large_order(folio)
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
    __slate_bits_0: __slate_bits::__SlateBits684U0,
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
    __slate_bits_0: __slate_bits::__SlateBits625U0,
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
fn memdesc_nid(mdf: *const __SlateRecord_mm_types_h_41) -> i32 {
ASSERT_EXCLUSIVE_BITS!((*mdf).f, 63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64));
((unsafe { (*mdf).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64) & 63u64) as i32
}

fn folio_nid(folio: *const folio) -> i32 {
memdesc_nid(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

fn folio_pgdat(folio: *const folio) -> *mut pglist_data {
NODE_DATA!(folio_nid(folio))
}

fn mk_pte(page: *const page, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
return pfn_pte(((unsafe { page.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64, pgprot);
}

fn folio_nr_pages(folio: *const folio) -> u64 {
if !folio_test_large(folio) {
return 1;
}
folio_large_nr_pages(folio)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_event_state {
    event: [u64; 75],
}


fn count_vm_event(mut item: u32) {
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
}

fn lruvec_stat_mod_folio(folio: *mut folio, idx: u32, val: i32) {
unsafe { mod_node_page_state(folio_pgdat(folio as *const folio), idx, val as i64) };
}

fn lruvec_stat_add_folio(folio: *mut folio, idx: u32) {
lruvec_stat_mod_folio(folio, idx, folio_nr_pages(folio as *const folio) as i32);
}

fn lruvec_stat_sub_folio(folio: *mut folio, idx: u32) {
lruvec_stat_mod_folio(folio, idx, folio_nr_pages(folio as *const folio).wrapping_neg() as i32);
}

#[inline(always)]
fn lowmem_page_address(page: *const page) -> *mut () {
page_to_virt!(page)
}

fn folio_address(folio: *const folio) -> *mut () {
page_address__2!(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) })
}

fn virt_to_ptdesc(x: *const ()) -> *mut ptdesc {
page_ptdesc!(((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__phys_addr_nodebug(x as u64) >> 12i32) as isize))
}

fn ptdesc_address(pt: *const ptdesc) -> *mut () {
folio_address(ptdesc_folio__2!(pt))
}

fn ptdesc_set_kernel(ptdesc: *mut ptdesc) {
set_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*ptdesc).pt_flags.f) }) as *mut u64);
}

fn ptdesc_clear_kernel(ptdesc: *mut ptdesc) {
// Note: the 'PG_referenced' bit does not strictly need to be
// cleared before freeing the page. But this is nice for
// symmetry.
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*ptdesc).pt_flags.f) }) as *mut u64);
}

fn ptdesc_test_kernel(ptdesc: *const ptdesc) -> bool {
test_bit__4!(2i32, unsafe { core::ptr::addr_of!((*ptdesc).pt_flags.f) }) != 0
}

fn pagetable_alloc_noprof(gfp: u32, order: u32) -> *mut ptdesc {
let page: *mut page = unsafe { alloc_pages_noprof(gfp | (BIT!(18i32) as u32), order) };
page_ptdesc!(page)
}

fn __pagetable_free(pt: *mut ptdesc) {
let page: *mut page = ptdesc_page!(pt);
unsafe { __free_pages(page, compound_order(page as *const page)) };
}

fn pagetable_free(pt: *mut ptdesc) {
if ptdesc_test_kernel(pt as *const ptdesc) {
ptdesc_clear_kernel(pt);
unsafe { pagetable_free_kernel(pt) };
} else {
__pagetable_free(pt);
}
}

fn ptlock_alloc(ptdesc: *mut ptdesc) -> bool {
true
}

fn ptlock_free(ptdesc: *mut ptdesc) {
}

fn ptlock_ptr(ptdesc: *mut ptdesc) -> *mut spinlock {
unsafe { core::ptr::addr_of_mut!((*ptdesc).__slate_anon_4.ptl) }
}

fn ptlock_init(ptdesc: *mut ptdesc) -> bool {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
// prep_new_page() initialize page->private (and therefore page->ptl)
// with 0. Make sure nobody took it in use in between.
//
// It can happen if arch try to use slab for page table allocation:
// slab code uses page->slab_cache, which share storage with page->ptl.
VM_BUG_ON_PAGE!();
if !ptlock_alloc(ptdesc) {
return false;
}
'__slate_break_1: loop {
spinlock_check(ptlock_ptr(ptdesc));
unsafe {
*ptlock_ptr(ptdesc) = *{
__slate_compound_0 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
true
}

fn __pagetable_ctor(ptdesc: *mut ptdesc) {
let folio: *mut folio = ptdesc_folio!(ptdesc);
__folio_set_pgtable(folio);
lruvec_stat_add_folio(folio, 38u32);
}

fn pagetable_dtor(ptdesc: *mut ptdesc) {
let folio: *mut folio = ptdesc_folio!(ptdesc);
ptlock_free(ptdesc);
__folio_clear_pgtable(folio);
lruvec_stat_sub_folio(folio, 38u32);
}

fn pagetable_dtor_free(ptdesc: *mut ptdesc) {
pagetable_dtor(ptdesc);
pagetable_free(ptdesc);
}

fn pagetable_pte_ctor(mm: *mut mm_struct, ptdesc: *mut ptdesc) -> bool {
let __v0: bool = mm != unsafe { core::ptr::addr_of_mut!(init_mm) } && !ptlock_init(ptdesc);
if __v0 {
return false;
}
__pagetable_ctor(ptdesc);
true
}

fn pmd_ptlock_init(ptdesc: *mut ptdesc) -> bool {
ptlock_init(ptdesc)
}

fn pagetable_pmd_ctor(mm: *mut mm_struct, ptdesc: *mut ptdesc) -> bool {
let __v0: bool = mm != unsafe { core::ptr::addr_of_mut!(init_mm) } && !pmd_ptlock_init(ptdesc);
if __v0 {
return false;
}
ptdesc_pmd_pts_init(ptdesc);
__pagetable_ctor(ptdesc);
true
}

fn pagetable_pud_ctor(ptdesc: *mut ptdesc) {
__pagetable_ctor(ptdesc);
}

fn pagetable_p4d_ctor(ptdesc: *mut ptdesc) {
__pagetable_ctor(ptdesc);
}

fn debug_pagealloc_enabled() -> bool {
IS_ENABLED!() != 0 && unsafe { _debug_pagealloc_enabled_early }
}

fn arch_is_platform_page(paddr: u64) -> bool {
false
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
fn kmap_flush_unused() {
}

// kunmap_local - Unmap a page mapped via kmap_local_page().
// @__addr: An address within the page mapped
//
// @__addr can be any address within the mapped page.  Commonly it is the
// address return from kmap_local_page(), but it can also include offsets.
//
// Unmapping should be done in the reverse order of the mapping.  See
// kmap_local_page() for details.
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
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
    file: *mut file,
    mapping: *mut address_space,
    ra: *mut file_ra_state,
    _index: u64,
    _nr_pages: u32,
    _batch_count: u32,
    dropbehind: bool,
    _workingset: bool,
    _pflags: u64,
}


fn __pte_alloc_one_kernel_noprof(mm: *mut mm_struct) -> *mut __SlateRecord_pgtable_64_types_h_21 {
let ptdesc: *mut ptdesc = pagetable_alloc_noprof(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0, 0u32);
if ptdesc.is_null() {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
if !pagetable_pte_ctor(mm, ptdesc) {
pagetable_free(ptdesc);
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
ptdesc_set_kernel(ptdesc);
ptdesc_address(ptdesc as *const ptdesc) as *mut __SlateRecord_pgtable_64_types_h_21
}

fn pte_alloc_one_kernel_noprof(mm: *mut mm_struct) -> *mut __SlateRecord_pgtable_64_types_h_21 {
__pte_alloc_one_kernel_noprof(mm)
}

fn pte_free_kernel(mm: *mut mm_struct, pte: *mut __SlateRecord_pgtable_64_types_h_21) {
pagetable_dtor_free(virt_to_ptdesc(pte as *const ()));
}

fn pmd_alloc_one_noprof(mm: *mut mm_struct, addr: u64) -> *mut __SlateRecord_pgtable_64_types_h_22 {
let mut gfp: u32 = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0 | (BIT!(22i32) as u32);
if mm == unsafe { core::ptr::addr_of_mut!(init_mm) } {
gfp = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0;
}
let ptdesc: *mut ptdesc = pagetable_alloc_noprof(gfp, 0u32);
if ptdesc.is_null() {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>();
}
if !pagetable_pmd_ctor(mm, ptdesc) {
pagetable_free(ptdesc);
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>();
}
if mm == unsafe { core::ptr::addr_of_mut!(init_mm) } {
ptdesc_set_kernel(ptdesc);
}
ptdesc_address(ptdesc as *const ptdesc) as *mut __SlateRecord_pgtable_64_types_h_22
}

fn pmd_free(mm: *mut mm_struct, pmd: *mut __SlateRecord_pgtable_64_types_h_22) {
let ptdesc: *mut ptdesc = virt_to_ptdesc(pmd as *const ());
loop {
if unlikely__3!((pmd as u64) & (_AC!() << 12i32).wrapping_sub(1)) != 0 {
loop {
unsafe {
core::arch::asm!("573: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 573b - ., 3; .popsection\n# {0}", const 573 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_pmd_free_0, sym __slate_asm_string_pmd_free_1, const 164 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
pagetable_dtor_free(ptdesc);
}

fn __pud_alloc_one_noprof(mm: *mut mm_struct, addr: u64) -> *mut __SlateRecord_pgtable_types_h_368 {
let mut gfp: u32 = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0 | (BIT!(22i32) as u32);
if mm == unsafe { core::ptr::addr_of_mut!(init_mm) } {
gfp = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0;
}
let ptdesc: *mut ptdesc = pagetable_alloc_noprof(gfp, 0u32);
if ptdesc.is_null() {
return core::ptr::null_mut::<__SlateRecord_pgtable_types_h_368>();
}
pagetable_pud_ctor(ptdesc);
if mm == unsafe { core::ptr::addr_of_mut!(init_mm) } {
ptdesc_set_kernel(ptdesc);
}
ptdesc_address(ptdesc as *const ptdesc) as *mut __SlateRecord_pgtable_types_h_368
}

fn pud_alloc_one_noprof(mm: *mut mm_struct, addr: u64) -> *mut __SlateRecord_pgtable_types_h_368 {
__pud_alloc_one_noprof(mm, addr)
}

fn __p4d_alloc_one_noprof(mm: *mut mm_struct, addr: u64) -> *mut __SlateRecord_pgtable_types_h_342 {
let mut gfp: u32 = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0 | (BIT!(22i32) as u32);
if mm == unsafe { core::ptr::addr_of_mut!(init_mm) } {
gfp = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32) | 0;
}
let ptdesc: *mut ptdesc = pagetable_alloc_noprof(gfp, 0u32);
if ptdesc.is_null() {
return core::ptr::null_mut::<__SlateRecord_pgtable_types_h_342>();
}
pagetable_p4d_ctor(ptdesc);
if mm == unsafe { core::ptr::addr_of_mut!(init_mm) } {
ptdesc_set_kernel(ptdesc);
}
ptdesc_address(ptdesc as *const ptdesc) as *mut __SlateRecord_pgtable_types_h_342
}

fn p4d_alloc_one_noprof(mm: *mut mm_struct, addr: u64) -> *mut __SlateRecord_pgtable_types_h_342 {
__p4d_alloc_one_noprof(mm, addr)
}

fn paravirt_alloc_pte(mm: *mut mm_struct, pfn: u64) {
}

/// The current flushing context - we pass it instead of 5 arguments:
#[repr(C)]
#[derive(Clone, Copy)]
struct cpa_data {
    vaddr: *mut u64,
    pgd: *mut __SlateRecord_pgtable_types_h_295,
    mask_set: pgprot,
    mask_clr: pgprot,
    numpages: u64,
    curpage: u64,
    pfn: u64,
    flags: u32,
    __slate_bits_0: __slate_bits::__SlateBits743U0,
    pages: *mut *mut page,
}


static cpa_warn_level: i32 = 1;

/// Serialize cpa() using cpa_lock so that we don't allow any other cpu, with
/// stale large tlb entries, to change the page attribute in parallel to some
/// other cpu splitting a large page entry along with changing the attribute.
static mut cpa_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

// Do not search for aliases
// try to collapse large pages
fn cachemode2pgprot(pcm: u32) -> pgprot {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = pgprot { pgprot: cachemode2protval(pcm) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

static mut direct_pages_count: __SlateAlign16<[u64; 6]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
extern "C" fn update_page_count(level: i32, pages: u64) {
// Protect against CPA
let mut __UNIQUE_ID_guard_585: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(pgd_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_585) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_586: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(pgd_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_586) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let __v2: *mut u64 = unsafe { core::ptr::addr_of_mut!(direct_pages_count.0) as *mut u64 }.wrapping_offset(level as isize);
let __v3: u64 = unsafe { *__v2 };
unsafe {
*__v2 = __v3.wrapping_add(pages);
}
}

fn split_page_count(level: i32) {
if (unsafe { direct_pages_count.0[level as usize] }) == 0 {
return;
}
let __v0: *mut u64 = unsafe { core::ptr::addr_of_mut!(direct_pages_count.0) as *mut u64 }.wrapping_offset(level as isize);
let __v1: u64 = unsafe { *__v0 };
unsafe {
*__v0 = __v1.wrapping_sub(1);
}
if ((unsafe { system_state }) as u32) == 3 {
if level == 2 {
count_vm_event(64u32);
} else {
if level == 3 {
count_vm_event(65u32);
}
}
}
let __v3: *mut u64 = unsafe { core::ptr::addr_of_mut!(direct_pages_count.0) as *mut u64 }.wrapping_offset(level.wrapping_sub(1) as isize);
let __v4: u64 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_add(512);
}
}

fn collapse_page_count(level: i32) {
let __v0: *mut u64 = unsafe { core::ptr::addr_of_mut!(direct_pages_count.0) as *mut u64 }.wrapping_offset(level as isize);
let __v1: u64 = unsafe { *__v0 };
unsafe {
*__v0 = __v1.wrapping_add(1);
}
if ((unsafe { system_state }) as u32) == 3 {
if level == 2 {
count_vm_event(66u32);
} else {
if level == 3 {
count_vm_event(67u32);
}
}
}
let __v3: *mut u64 = unsafe { core::ptr::addr_of_mut!(direct_pages_count.0) as *mut u64 }.wrapping_offset(level.wrapping_sub(1) as isize);
let __v4: u64 = unsafe { *__v3 };
unsafe {
*__v3 = __v4.wrapping_sub(512);
}
}

#[unsafe(no_mangle)]
extern "C" fn arch_report_meminfo(m: *mut seq_file) {
unsafe { seq_printf(m, c"DirectMap4k:    %8lu kB\n".as_ptr() as *const u8, direct_pages_count.0[1] << 2i32) };
unsafe { seq_printf(m, c"DirectMap2M:    %8lu kB\n".as_ptr() as *const u8, direct_pages_count.0[2] << 11i32) };
if (unsafe { direct_gbpages }) != 0 {
unsafe { seq_printf(m, c"DirectMap1G:    %8lu kB\n".as_ptr() as *const u8, direct_pages_count.0[3] << 20i32) };
}
}

fn cpa_inc_1g_checked() {
}

fn cpa_inc_2m_checked() {
}

fn cpa_inc_4k_install() {
}

fn cpa_inc_lp_sameprot(level: i32) {
}

fn cpa_inc_lp_preserved(level: i32) {
}

fn within(addr: u64, start: u64, end: u64) -> i32 {
(addr >= start && addr < end) as i32
}

fn within_inclusive(addr: u64, start: u64, end: u64) -> i32 {
(addr >= start && addr <= end) as i32
}

// The kernel image is mapped into two places in the virtual address space
// (addresses without KASLR, of course):
//
// 1. The kernel direct map (0xffff880000000000)
// 2. The "high kernel map" (0xffffffff81000000)
//
// We actually execute out of #2. If we get the address of a kernel symbol, it
// points to #2, but almost all physical-to-virtual translations point to #1.
//
// This is so that we can have both a directmap of all physical memory *and*
// take full advantage of the limited (s32) immediate addressing range (2G)
// of x86_64.
//
// See Documentation/arch/x86/x86_64/mm.rst for more detail.
fn highmap_start_pfn() -> u64 {
__pa_symbol!(unsafe { core::ptr::addr_of_mut!(_text) as *mut u8 }) >> 12i32
}

fn highmap_end_pfn() -> u64 {
// Do not reference physical address outside the kernel.
__pa_symbol!(roundup!(unsafe { _brk_end }, 2097152u64).wrapping_sub(1)) >> 12i32
}

fn __cpa_pfn_in_highmap(pfn: u64) -> bool {
// Kernel text has an alias mapping at a high address, known
// here as "highmap".
within_inclusive(pfn, highmap_start_pfn(), highmap_end_pfn()) != 0
}

/// See set_mce_nospec().
///
/// Machine check recovery code needs to change cache mode of poisoned pages to
/// UC to avoid speculative access logging another error. But passing the
/// address of the 1:1 mapping to set_memory_uc() is a fine way to encourage a
/// speculative access. So we cheat and flip the top bit of the address. This
/// works fine for the code that updates the page tables. But at the end of the
/// process we need to flush the TLB and cache and the non-canonical address
/// causes a #GP fault when used by the INVLPG and CLFLUSH instructions.
///
/// But in the common case we already have a canonical address. This code
/// will fix the top bit if needed and is a no-op otherwise.
fn fix_addr(addr: u64) -> u64 {
(((addr << 1i32) as i64) >> 1i32) as u64
}

fn __cpa_addr(cpa: *mut cpa_data, idx: u64) -> u64 {
if (unsafe { (*cpa).flags }) & 4 != 0 {
let page: *mut page = unsafe { *unsafe { (*cpa).pages }.wrapping_offset(idx as isize) };
if unlikely!(PageHighMem(page as *const page)) != 0 {
return 0;
}
return page_address!(page) as u64;
}
if (unsafe { (*cpa).flags }) & 2 != 0 {
return unsafe { *unsafe { (*cpa).vaddr }.wrapping_offset(idx as isize) };
}
unsafe { *(*cpa).vaddr }.wrapping_add(idx.wrapping_mul(_AC!() << 12i32))
}

// Flushing functions
fn clflush_cache_range_opt(vaddr: *mut (), size: u32) {
let clflush_size: u64 = (unsafe { boot_cpu_data.x86_clflush_size }) as u64;
let mut p: *mut () = ((vaddr as u64) & !clflush_size.wrapping_sub(1)) as *mut ();
let vend: *mut () = (unsafe { (vaddr as *const u8).wrapping_offset(size as isize) }) as *mut ();
if p >= vend {
return;
}
while p < vend {
clflushopt(p as *mut ());
p = (unsafe { (p as *const u8).wrapping_offset(clflush_size as isize) }) as *mut ();
}
}

/// clflush_cache_range - flush a cache range with clflush
/// @vaddr:	virtual start address
/// @size:	number of bytes to flush
///
/// CLFLUSHOPT is an unordered instruction which needs fencing with MFENCE or
/// SFENCE to avoid ordering issues.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.clflush_cache_range")]
extern "C" fn clflush_cache_range(vaddr: *mut (), size: u32) {
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
clflush_cache_range_opt(vaddr, size);
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_clflush_cache_range_587: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), u32)>, *mut ()>(Some(clflush_cache_range)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.arch_invalidate_pmem")]
extern "C" fn arch_invalidate_pmem(addr: *mut (), size: u64) {
clflush_cache_range(addr, size as u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_arch_invalidate_pmem_588: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), u64)>, *mut ()>(Some(arch_invalidate_pmem)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.cpu_cache_has_invalidate_memregion")]
extern "C" fn cpu_cache_has_invalidate_memregion() -> bool {
cpu_feature_enabled__5!(159i32) == 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_cpu_cache_has_invalidate_memregion_589: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> bool>, *mut ()>(Some(cpu_cache_has_invalidate_memregion)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.cpu_cache_invalidate_memregion")]
extern "C" fn cpu_cache_invalidate_memregion(start: u64, len: u64) -> i32 {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!cpu_cache_has_invalidate_memregion())) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("590: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 590b - ., 3; .popsection\n# {0}", const 590 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cpu_cache_invalidate_memregion_1, sym __slate_asm_string_cpu_cache_invalidate_memregion_2, const 374 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("591: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 591b - ., 4; .popsection\n# {0}", const 591 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 6i32.wrapping_neg();
}
unsafe { wbinvd_on_all_cpus() };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_cpu_cache_invalidate_memregion_592: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u64) -> i32>, *mut ()>(Some(cpu_cache_invalidate_memregion)) };

#[unsafe(link_section = ".text.slate_distinct.set_memory.__cpa_flush_all")]
extern "C" fn __cpa_flush_all(arg: *mut ()) {
let cache: u64 = arg as u64;
// Flush all to work around Errata in early athlons regarding
// large page flushing.
unsafe { __flush_tlb_all() };
if cache != 0 && ((unsafe { boot_cpu_data.__slate_anon_0.__slate_anon_0.x86 }) as i32) >= 4 {
wbinvd();
}
}

fn cpa_flush_all(cache: u64) {
loop {
if unlikely__2!(irqs_disabled!() != 0 && !unsafe { early_boot_irqs_disabled }) != 0 {
loop {
unsafe {
core::arch::asm!("593: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 593b - ., 3; .popsection\n# {0}", const 593 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpa_flush_all_10, sym __slate_asm_string_cpa_flush_all_11, const 398 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
on_each_cpu(Some(__cpa_flush_all), cache as *mut (), 1i32);
}

#[unsafe(link_section = ".text.slate_distinct.set_memory.__cpa_flush_tlb")]
extern "C" fn __cpa_flush_tlb(data: *mut ()) {
let cpa: *mut cpa_data = data as *mut cpa_data;
let mut i: u32 = 0;
while (i as u64) < unsafe { (*cpa).numpages } {
unsafe { flush_tlb_one_kernel(fix_addr(__cpa_addr(cpa, i as u64))) };
i = i.wrapping_add(1);
}
}

fn __cpa_collapse_large_pages(cpa: *mut cpa_data) {
let mut start: u64 = 0;
let mut addr: u64 = 0;
let mut end: u64 = 0;
let mut ptdesc: *mut ptdesc = unsafe { core::mem::zeroed() };
let mut tmp: *mut ptdesc = unsafe { core::mem::zeroed() };
let mut pgtables: list_head = unsafe { core::mem::zeroed() };
pgtables = list_head { next: core::ptr::addr_of_mut!(pgtables), prev: core::ptr::addr_of_mut!(pgtables) };
let mut collapsed: i32 = 0;
let mut i: i32 = 0;
let mut __UNIQUE_ID_guard_594: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(cpa_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_594) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_595: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(cpa_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_595) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*cpa).flags }) & 6u32 != 0 {
i = 0;
while (i as u64) < unsafe { (*cpa).numpages } {
let __v2: i32 = collapse_large_pages(__cpa_addr(cpa, i as u64), core::ptr::addr_of_mut!(pgtables));
collapsed = collapsed.wrapping_add(__v2);
i = i.wrapping_add(1);
}
} else {
addr = __cpa_addr(cpa, 0u64);
start = addr & !(_AC!() << 21i32).wrapping_sub(1);
end = addr.wrapping_add((_AC!() << 12i32).wrapping_mul(unsafe { (*cpa).numpages }));
addr = start;
while within(addr, start, end) != 0 {
let __v8: i32 = collapse_large_pages(addr, core::ptr::addr_of_mut!(pgtables));
collapsed = collapsed.wrapping_add(__v8);
let __v12: u64 = addr;
addr = __v12.wrapping_add(_AC!() << 21i32);
}
}
if collapsed == 0 {
return;
}
unsafe { flush_tlb_all() };
ptdesc = list_first_entry__11!(core::ptr::addr_of_mut!(pgtables));
tmp = list_next_entry__11!(ptdesc, pt_list);
while list_entry_is_head__2!(ptdesc, core::ptr::addr_of_mut!(pgtables), pt_list) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*ptdesc).__slate_anon_1.pt_list) });
// Only early alloc'd direct map should not be flagged PG_table
// here and those shouldn't be collapsed. However be abundantly
// cautious and handle the !PG_table case too.
if PageTable(ptdesc_page!(ptdesc) as *const page) != 0 {
pagetable_dtor_free(ptdesc);
} else {
pagetable_free(ptdesc);
}
ptdesc = tmp;
tmp = list_next_entry__11!(tmp, pt_list);
}
}

fn cpa_collapse_large_pages(cpa: *mut cpa_data) {
// Take the mmap write lock on init_mm to:
// - Avoid a use-after-free if raced by ptdump (which takes its own
//   write lock on init_mm).
// - Serialise concurrent CPA walkers.
let mut scope: core::mem::MaybeUninit<*mut mm_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut mm_struct = core::ptr::addr_of_mut!(scope) as *mut *mut mm_struct;
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_mmap_write_lock_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
core::ptr::write(__slate_slot_0, class_mmap_write_lock_constructor__const7(core::ptr::addr_of_mut!(init_mm)));
__slate_cleanup_armed_0.set(true);
if !class_mmap_write_lock_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_mmap_write_lock_is_conditional {
if true {
__cpa_collapse_large_pages(cpa);
}
}
__slate_cleanup_armed_0.set(false);
class_mmap_write_lock_destructor(__slate_slot_0 as *mut _);
}
}

fn cpa_flush(cpa: *mut cpa_data, cache: i32) {
// Only flush present addresses:
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut pte: *mut __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
let mut level: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(level) as *mut u32;
let mut addr: u64 = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
unsafe {
'__slate_dispatch: {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
'__join_24: {
loop {
if unlikely__2!(irqs_disabled!() != 0 && !early_boot_irqs_disabled) != 0 {
break '__join_24;
} else {
if true {
break;
}
}
}
if cache != 0 {
__v6 = cpu_feature_enabled__6!(19i32) == 0;
} else {
__v6 = false;
}
if __v6 {
cpa_flush_all(cache as u64);
} else {
if ((*cpa).__slate_bits_0.__get_force_flush_all() as i32) != 0 || (*cpa).numpages > tlb_single_page_flush_ceiling {
flush_tlb_all();
} else {
on_each_cpu(Some(__cpa_flush_tlb), cpa as *mut (), 1i32);
}
if cache != 0 {
loop {
kcsan_mb!();
core::arch::asm!("mfence", options(nostack, att_syntax));
if true {
break;
}
}
*__slate_slot_7 = 0;
while (*__slate_slot_7 as u64) < (*cpa).numpages {
addr = __cpa_addr(cpa, *__slate_slot_7 as u64);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
pte = lookup_address(addr, core::ptr::addr_of_mut!(*__slate_slot_4));
if !pte.is_null() {
__v2 = pte_val!(unsafe { *pte }) & _AT!(u64, 1i32) << 0i32 != 0;
} else {
__v2 = false;
}
if __v2 {
clflush_cache_range_opt(fix_addr(addr) as *mut (), (_AC!() << 12i32) as u32);
}
let __v1: u32 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_add(1);
}
loop {
kcsan_mb!();
core::arch::asm!("mfence", options(nostack, att_syntax));
if true {
break;
}
}
}
}
if (*cpa).flags & 16 != 0 {
cpa_collapse_large_pages(cpa);
}
break '__slate_dispatch;
}
core::arch::asm!("597: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 597b - ., 3; .popsection\n# {0}", const 597 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_cpa_flush_10, sym __slate_asm_string_cpa_flush_11, const 472 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}

fn overlaps(r1_start: u64, r1_end: u64, r2_start: u64, r2_end: u64) -> bool {
r1_start <= r2_end && r1_end >= r2_start || r2_start <= r1_end && r2_end >= r1_start
}

fn protect_pci_bios(spfn: u64, epfn: u64) -> u64 {
0
}

/// The .rodata section needs to be read-only. Using the pfn catches all
/// aliases.  This also includes __ro_after_init, so do not enforce until
/// kernel_set_to_readonly is true.
fn protect_rodata(spfn: u64, epfn: u64) -> u64 {
let mut epfn_ro: u64 = 0;
let spfn_ro: u64 = PFN_DOWN!(__pa_symbol!(unsafe { core::ptr::addr_of_mut!(__start_rodata) as *mut u8 }));
// Note: __end_rodata is at page aligned and not inclusive, so
// subtract 1 to get the last enforced PFN in the rodata area.
epfn_ro = PFN_DOWN!(__pa_symbol!(unsafe { core::ptr::addr_of_mut!(__end_rodata) as *mut u8 })).wrapping_sub(1);
let __v0: bool = (unsafe { kernel_set_to_readonly }) != 0 && overlaps(spfn, epfn, spfn_ro, epfn_ro);
if __v0 {
return _AT!(u64, 1i32) << 1i32;
}
0
}

/// Protect kernel text against becoming non executable by forbidding
/// _PAGE_NX.  This protects only the high kernel mapping (_text -> _etext)
/// out of which the kernel actually executes.  Do not protect the low
/// mapping.
///
/// This does not cover __inittext since that is gone after boot.
fn protect_kernel_text(start: u64, end: u64) -> u64 {
let t_end: u64 = ((unsafe { core::ptr::addr_of_mut!(_etext) as *mut u8 }) as u64).wrapping_sub(1);
let t_start: u64 = (unsafe { core::ptr::addr_of_mut!(_text) as *mut u8 }) as u64;
if overlaps(start, end, t_start, t_end) {
return _AT!(u64, 1i32) << 63i32;
}
0
}

/// Once the kernel maps the text as RO (kernel_set_to_readonly is set),
/// kernel text mappings for the large page aligned text, rodata sections
/// will be always read-only. For the kernel identity mappings covering the
/// holes caused by this alignment can be anything that user asks.
///
/// This will preserve the large page mappings for kernel text/data at no
/// extra cost.
fn protect_kernel_text_ro(start: u64, end: u64) -> u64 {
let t_end: u64 = ((unsafe { core::ptr::addr_of_mut!(__end_rodata_hpage_align) as *mut u8 }) as u64).wrapping_sub(1);
let t_start: u64 = (unsafe { core::ptr::addr_of_mut!(_text) as *mut u8 }) as u64;
let mut level: u32 = 0;
let __v0: bool = (unsafe { kernel_set_to_readonly }) == 0 || !overlaps(start, end, t_start, t_end);
if __v0 {
return 0;
}
// Don't enforce the !RW mapping for the kernel text mapping, if
// the current mapping is already using small page mapping.  No
// need to work hard to preserve large page mappings in this case.
//
// This also fixes the Linux Xen paravirt guest boot failure caused
// by unexpected read-only mappings for kernel identity
// mappings. In this paravirt guest case, the kernel text mapping
// and the kernel identity mapping share the same page-table pages,
// so the protections for kernel text and identity mappings have to
// be the same.
if !lookup_address(start, core::ptr::addr_of_mut!(level)).is_null() && level != 1 {
return _AT!(u64, 1i32) << 1i32;
}
0
}

fn conflicts(prot: pgprot, val: u64) -> bool {
pgprot_val!(prot) & !val != pgprot_val!(prot)
}

fn check_conflict(warnlvl: i32, prot: pgprot, val: u64, start: u64, end: u64, pfn: u64, txt: *const u8) {
let __v0: bool = warnlvl > unsafe { cpa_warn_level } || !conflicts(prot, val);
if __v0 {
return;
}
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014CPA %8s %10s: 0x%016lx - 0x%016lx PFN %lx req %016llx prevent %016llx\n".as_ptr() as *const u8, lvltxt.0[warnlvl as usize], txt, start, end, pfn, pgprot_val!(prot), val) };
}

static mut lvltxt: __SlateAlign16<[*const u8; 3]> = __SlateAlign16([(b"conflict\0".as_ptr() as *mut u8) as *const u8, (b"protect\0".as_ptr() as *mut u8) as *const u8, (b"detect\0".as_ptr() as *mut u8) as *const u8]);

/// Certain areas of memory on x86 require very specific protection flags,
/// for example the BIOS area or kernel text. Callers don't always get this
/// right (again, ioremap() on BIOS memory is not uncommon) so this function
/// checks and fixes these known static required protection bits.
fn static_protections(prot: pgprot, start: u64, pfn: u64, npg: u64, lpsize: u64, warnlvl: i32) -> pgprot {
let mut __slate_compound_6: pgprot = unsafe { core::mem::zeroed() };
let mut forbidden: u64 = 0;
let mut res: u64 = 0;
let mut end: u64 = 0;
// There is no point in checking RW/NX conflicts when the requested
// mapping is setting the page !PRESENT.
if pgprot_val!(prot) & _AT!(u64, 1i32) << 0i32 == 0 {
return prot;
}
// Operate on the virtual address
end = start.wrapping_add(npg.wrapping_mul(_AC!() << 12i32)).wrapping_sub(1);
res = protect_kernel_text(start, end);
check_conflict(warnlvl, prot, res, start, end, pfn, c"Text NX".as_ptr() as *const u8);
forbidden = res;
// Special case to preserve a large page. If the change spawns the
// full large page mapping then there is no point to split it
// up. Happens with ftrace and is going to be removed once ftrace
// switched to text_poke().
if lpsize != npg.wrapping_mul(_AC!() << 12i32) || start & lpsize.wrapping_sub(1) != 0 {
res = protect_kernel_text_ro(start, end);
check_conflict(warnlvl, prot, res, start, end, pfn, c"Text RO".as_ptr() as *const u8);
forbidden |= res;
}
// Check the PFN directly
res = protect_pci_bios(pfn, pfn.wrapping_add(npg).wrapping_sub(1));
check_conflict(warnlvl, prot, res, start, end, pfn, c"PCIBIOS NX".as_ptr() as *const u8);
forbidden |= res;
res = protect_rodata(pfn, pfn.wrapping_add(npg).wrapping_sub(1));
check_conflict(warnlvl, prot, res, start, end, pfn, c"Rodata RO".as_ptr() as *const u8);
forbidden |= res;
return unsafe { *{
__slate_compound_6 = pgprot { pgprot: pgprot_val!(prot) & !forbidden };
core::ptr::addr_of_mut!(__slate_compound_6)
} };
}

/// Validate strict W^X semantics.
fn verify_rwx(old: pgprot, new: pgprot, start: u64, pfn: u64, npg: u64, nx: bool, rw: bool) -> pgprot {
let mut end: u64 = 0;
// 32-bit has some unfixable W+X issues, like EFI code
// and writeable data being in the same page.  Disable
// detection and enforcement there.
if IS_ENABLED!() != 0 {
return new;
}
// Only verify when NX is supported:
if (unsafe { __supported_pte_mask }) & _AT!(u64, 1i32) << 63i32 == 0 {
return new;
}
if (pgprot_val!(old) ^ pgprot_val!(new)) & (_AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 63i32) == 0 {
return new;
}
if pgprot_val!(new) & (_AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 63i32) != _AT!(u64, 1i32) << 1i32 {
return new;
}
// Non-leaf translation entries can disable writing or execution.
if !rw || nx {
return new;
}
end = start.wrapping_add(npg.wrapping_mul(_AC!() << 12i32)).wrapping_sub(1);
let mut __v0: i32;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"CPA detected W^X violation: %016llx -> %016llx range: 0x%016lx - 0x%016lx PFN %lx\n".as_ptr() as *const u8, pgprot_val!(old), pgprot_val!(new), start, end, pfn) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_verify_rwx_3, sym __slate_asm_string_verify_rwx_4, const 725 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, pgprot_val!(old), pgprot_val!(new), start, end, pfn) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = __ret_warn_on;
// For now, allow all permission change attempts by returning the
// attempted permissions.  This can 'return old' to actively
// refuse the permission change at a later time.
return new;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_598: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Lookup the page table entry for a virtual address in a specific pgd.
/// Return a pointer to the entry (or NULL if the entry does not exist),
/// the level of the entry, and the effective NX and RW bits of all
/// page table levels.
#[unsafe(no_mangle)]
extern "C" fn lookup_address_in_pgd_attr(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64, level: *mut u32, nx: *mut bool, rw: *mut bool) -> *mut __SlateRecord_pgtable_64_types_h_21 {
unsafe {
*level = 5;
}
unsafe {
*nx = false;
}
unsafe {
*rw = true;
}
if pgd_none(unsafe { *pgd }) != 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
unsafe {
*level = 4;
}
let __v0: u64 = pgd_flags(unsafe { *pgd }) & _AT!(u64, 1i32) << 63i32;
unsafe {
*nx = ((*nx as i32) as u64) | __v0 != 0;
}
let __v4: u64 = pgd_flags(unsafe { *pgd }) & _AT!(u64, 1i32) << 1i32;
unsafe {
*rw = ((*rw as i32) as u64) & __v4 != 0;
}
let p4d: *mut __SlateRecord_pgtable_types_h_342 = p4d_offset(pgd, address);
if p4d_none(unsafe { *p4d }) != 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
let __v8: bool = p4d_leaf!() != 0 || p4d_present(unsafe { *p4d }) == 0;
if __v8 {
return p4d as *mut __SlateRecord_pgtable_64_types_h_21;
}
unsafe {
*level = 3;
}
let __v9: u64 = p4d_flags(unsafe { *p4d }) & _AT!(u64, 1i32) << 63i32;
unsafe {
*nx = ((*nx as i32) as u64) | __v9 != 0;
}
let __v13: u64 = p4d_flags(unsafe { *p4d }) & _AT!(u64, 1i32) << 1i32;
unsafe {
*rw = ((*rw as i32) as u64) & __v13 != 0;
}
let pud: *mut __SlateRecord_pgtable_types_h_368 = pud_offset(p4d, address);
if pud_none(unsafe { *pud }) != 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
let __v17: bool = pud_leaf(unsafe { *pud }) || pud_present(unsafe { *pud }) == 0;
if __v17 {
return pud as *mut __SlateRecord_pgtable_64_types_h_21;
}
unsafe {
*level = 2;
}
let __v18: u64 = pud_flags(unsafe { *pud }) & _AT!(u64, 1i32) << 63i32;
unsafe {
*nx = ((*nx as i32) as u64) | __v18 != 0;
}
let __v22: u64 = pud_flags(unsafe { *pud }) & _AT!(u64, 1i32) << 1i32;
unsafe {
*rw = ((*rw as i32) as u64) & __v22 != 0;
}
let pmd: *mut __SlateRecord_pgtable_64_types_h_22 = pmd_offset(pud, address);
if pmd_none(unsafe { *pmd }) != 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
}
let __v26: bool = pmd_leaf(unsafe { *pmd }) || pmd_present(unsafe { *pmd }) == 0;
if __v26 {
return pmd as *mut __SlateRecord_pgtable_64_types_h_21;
}
unsafe {
*level = 1;
}
let __v27: u64 = pmd_flags(unsafe { *pmd }) & _AT!(u64, 1i32) << 63i32;
unsafe {
*nx = ((*nx as i32) as u64) | __v27 != 0;
}
let __v31: u64 = pmd_flags(unsafe { *pmd }) & _AT!(u64, 1i32) << 1i32;
unsafe {
*rw = ((*rw as i32) as u64) & __v31 != 0;
}
pte_offset_kernel(pmd, address)
}

/// Lookup the page table entry for a virtual address in a specific pgd.
/// Return a pointer to the entry and the level of the mapping.
#[unsafe(no_mangle)]
extern "C" fn lookup_address_in_pgd(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64, level: *mut u32) -> *mut __SlateRecord_pgtable_64_types_h_21 {
let mut nx: bool = false;
let mut rw: bool = false;
lookup_address_in_pgd_attr(pgd, address, level, core::ptr::addr_of_mut!(nx), core::ptr::addr_of_mut!(rw))
}

/// Lookup the page table entry for a virtual address. Return a pointer
/// to the entry and the level of the mapping.
///
/// Note: the function returns p4d, pud or pmd either when the entry is marked
/// large or when the present bit is not set. Otherwise it returns NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.lookup_address")]
extern "C" fn lookup_address(address: u64, level: *mut u32) -> *mut __SlateRecord_pgtable_64_types_h_21 {
lookup_address_in_pgd(pgd_offset_k!(address), address, level)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_lookup_address_599: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, *mut u32) -> *mut __SlateRecord_pgtable_64_types_h_21>, *mut ()>(Some(lookup_address)) };

fn _lookup_address_cpa(cpa: *mut cpa_data, address: u64, level: *mut u32, nx: *mut bool, rw: *mut bool) -> *mut __SlateRecord_pgtable_64_types_h_21 {
let mut pgd: *mut __SlateRecord_pgtable_types_h_295 = unsafe { core::mem::zeroed() };
if unsafe { (*cpa).pgd }.is_null() {
pgd = pgd_offset_k!(address);
} else {
pgd = unsafe { (*cpa).pgd }.wrapping_offset(pgd_index!(address) as isize);
}
lookup_address_in_pgd_attr(pgd, address, level, nx, rw)
}

/// Lookup the PMD entry for a virtual address. Return a pointer to the entry
/// or NULL if not present.
#[unsafe(no_mangle)]
extern "C" fn lookup_pmd_address(address: u64) -> *mut __SlateRecord_pgtable_64_types_h_22 {
let pgd: *mut __SlateRecord_pgtable_types_h_295 = pgd_offset_k!(address);
if pgd_none(unsafe { *pgd }) != 0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>();
}
let p4d: *mut __SlateRecord_pgtable_types_h_342 = p4d_offset(pgd, address);
let __v0: bool = p4d_none(unsafe { *p4d }) != 0 || p4d_leaf!() != 0 || p4d_present(unsafe { *p4d }) == 0;
if __v0 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>();
}
let pud: *mut __SlateRecord_pgtable_types_h_368 = pud_offset(p4d, address);
let __v1: bool = pud_none(unsafe { *pud }) != 0 || pud_leaf(unsafe { *pud });
let __v2: bool = __v1 || pud_present(unsafe { *pud }) == 0;
if __v2 {
return core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>();
}
pmd_offset(pud, address)
}

/// This is necessary because __pa() does not work on some
/// kinds of memory, like vmalloc() or the alloc_remap()
/// areas on 32-bit NUMA systems.  The percpu areas can
/// end up in this kind of memory, for instance.
///
/// Note that as long as the PTEs are well-formed with correct PFNs, this
/// works without checking the PRESENT bit in the leaf PTE.  This is unlike
/// the similar vmalloc_to_page() and derivatives.  Callers may depend on
/// this behavior.
///
/// This could be optimized, but it is only used in paths that are not perf
/// sensitive, and keeping it unoptimized should increase the testing coverage
/// for the more obscure platforms.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.slow_virt_to_phys")]
extern "C" fn slow_virt_to_phys(__virt_addr: *mut ()) -> u64 {
let virt_addr: u64 = __virt_addr as u64;
let mut phys_addr: u64 = 0;
let mut offset: u64 = 0;
let mut level: u32 = 0;
let pte: *mut __SlateRecord_pgtable_64_types_h_21 = lookup_address(virt_addr, core::ptr::addr_of_mut!(level) as *mut u32);
loop {
if unlikely__2!(pte.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("600: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 600b - ., 3; .popsection\n# {0}", const 600 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_slow_virt_to_phys_0, sym __slate_asm_string_slow_virt_to_phys_1, const 882 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if true {
break;
}
}
// pXX_pfn() returns unsigned long, which must be cast to phys_addr_t
// before being left-shifted PAGE_SHIFT bits -- this trick is to
// make 32-PAE kernel work correctly.
match level as u32 {
3 => {
phys_addr = pud_pfn(unsafe { *(pte as *mut __SlateRecord_pgtable_types_h_368) }) << 12i32;
offset = virt_addr & !(!(_AC!() << 30i32).wrapping_sub(1));
}
2 => {
phys_addr = pmd_pfn(unsafe { *(pte as *mut __SlateRecord_pgtable_64_types_h_22) }) << 12i32;
offset = virt_addr & !(!(_AC!() << 21i32).wrapping_sub(1));
}
_ => {
phys_addr = pte_pfn(unsafe { *pte }) << 12i32;
offset = virt_addr & !(!(_AC!() << 12i32).wrapping_sub(1));
}
}
phys_addr | offset
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_slow_virt_to_phys_601: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ()) -> u64>, *mut ()>(Some(slow_virt_to_phys)) };

/// Set the new pmd in all the pgds we know about:
fn __set_pmd_pte(kpte: *mut __SlateRecord_pgtable_64_types_h_21, address: u64, pte: __SlateRecord_pgtable_64_types_h_21) {
// change init_mm
set_pte_atomic!(kpte, pte);
}

fn pgprot_clear_protnone_bits(mut prot: pgprot) -> pgprot {
// _PAGE_GLOBAL means "global page" for present PTEs.
// But, it is also used to indicate _PAGE_PROTNONE
// for non-present PTEs.
//
// This ensures that a _PAGE_GLOBAL PTE going from
// present to non-present is not confused as
// _PAGE_PROTNONE.
if pgprot_val!(prot) & _AT!(u64, 1i32) << 0i32 == 0 {
let __v0: u64 = prot.pgprot;
let __v1: u64 = __v0 & !(_AT!(u64, 1i32) << 8i32);
prot.pgprot = __v1;
}
return prot;
}

fn __should_split_large_page(kpte: *mut __SlateRecord_pgtable_64_types_h_21, address: u64, cpa: *mut cpa_data) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: pgprot = unsafe { core::mem::zeroed() };
let mut numpages: u64 = 0;
let mut pmask: u64 = 0;
let mut psize: u64 = 0;
let mut lpaddr: u64 = 0;
let mut pfn: u64 = 0;
let mut old_pfn: u64 = 0;
let mut old_prot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(old_prot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut new_prot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new_prot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut req_prot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(req_prot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut chk_prot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(chk_prot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut new_pte: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(new_pte) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut level: u32 = 0;
let mut nx: bool = false;
let mut rw: bool = false;
// Check for races, another CPU might have split this page
// up already:
let tmp: *mut __SlateRecord_pgtable_64_types_h_21 = _lookup_address_cpa(cpa, address, core::ptr::addr_of_mut!(level) as *mut u32, core::ptr::addr_of_mut!(nx), core::ptr::addr_of_mut!(rw));
if tmp != kpte {
return 1;
}
match level as u32 {
2 => {
old_prot = unsafe { *{
__slate_compound_0 = pgprot { pgprot: pmd_flags(*(kpte as *mut __SlateRecord_pgtable_64_types_h_22)) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
old_pfn = pmd_pfn(unsafe { *(kpte as *mut __SlateRecord_pgtable_64_types_h_22) });
cpa_inc_2m_checked();
}
3 => {
old_prot = unsafe { *{
__slate_compound_2 = pgprot { pgprot: pud_flags(*(kpte as *mut __SlateRecord_pgtable_types_h_368)) };
core::ptr::addr_of_mut!(__slate_compound_2)
} };
old_pfn = pud_pfn(unsafe { *(kpte as *mut __SlateRecord_pgtable_types_h_368) });
cpa_inc_1g_checked();
}
_ => {
return 22i32.wrapping_neg();
}
}
psize = page_level_size(level);
pmask = page_level_mask(level);
// Calculate the number of pages, which fit into this large
// page starting at address:
lpaddr = address.wrapping_add(psize) & pmask;
numpages = lpaddr.wrapping_sub(address) >> 12i32;
if numpages < unsafe { (*cpa).numpages } {
unsafe {
(*cpa).numpages = numpages;
}
}
// We are safe now. Check whether the new pgprot is the same:
// Convert protection attributes to 4k-format, as cpa->mask* are set
// up accordingly.
// Clear PSE (aka _PAGE_PAT) and move PAT bit to correct position
req_prot = pgprot_large_2_4k(old_prot);
let __v3: u64 = req_prot.pgprot;
let __v4: u64 = __v3 & !pgprot_val__2!((*cpa).mask_clr);
req_prot.pgprot = __v4;
let __v5: u64 = req_prot.pgprot;
let __v6: u64 = __v5 | pgprot_val__2!((*cpa).mask_set);
req_prot.pgprot = __v6;
// req_prot is in format of 4k pages. It must be converted to large
// page format: the caching mode includes the PAT bit located at
// different bit positions in the two formats.
req_prot = pgprot_4k_2_large(req_prot);
req_prot = pgprot_clear_protnone_bits(req_prot);
if pgprot_val!(req_prot) & _AT!(u64, 1i32) << 0i32 != 0 {
let __v7: u64 = req_prot.pgprot;
let __v8: u64 = __v7 | _AT!(u64, 1i32) << 7i32;
req_prot.pgprot = __v8;
}
// old_pfn points to the large page base pfn. So we need to add the
// offset of the virtual address:
pfn = old_pfn.wrapping_add((address & psize.wrapping_sub(1)) >> 12i32);
unsafe {
(*cpa).pfn = pfn;
}
// Calculate the large page base address and the number of 4K pages
// in the large page
lpaddr = address & pmask;
numpages = psize >> 12i32;
// Sanity check that the existing mapping is correct versus the static
// protections. static_protections() guards against !PRESENT, so no
// extra conditional required here.
chk_prot = static_protections(old_prot, lpaddr, old_pfn, numpages, psize, 0i32);
let mut __v9: i64;
let __ret_warn_on: i32 = (pgprot_val!(chk_prot) != pgprot_val!(old_prot)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("602: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 602b - ., 3; .popsection\n# {0}", const 602 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___should_split_large_page_10, sym __slate_asm_string___should_split_large_page_11, const 1038 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("603: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 603b - ., 4; .popsection\n# {0}", const 603 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v9 = (__ret_warn_on != 0) as i64;
if __v9 != 0 {
// Split the large page and tell the split code to
// enforce static protections.
unsafe {
(*cpa).__slate_bits_0.__set_force_static_prot(1u32);
}
return 1;
}
// Optimization: If the requested pgprot is the same as the current
// pgprot, then the large page can be preserved and no updates are
// required independent of alignment and length of the requested
// range. The above already established that the current pgprot is
// correct, which in consequence makes the requested pgprot correct
// as well if it is the same. The static protection scan below will
// not come to a different conclusion.
if pgprot_val!(req_prot) == pgprot_val!(old_prot) {
cpa_inc_lp_sameprot((level as u32) as i32);
return 0;
}
// If the requested range does not cover the full page, split it up
if address != lpaddr || (unsafe { (*cpa).numpages }) != numpages {
return 1;
}
// Check whether the requested pgprot is conflicting with a static
// protection requirement in the large page.
new_prot = static_protections(req_prot, lpaddr, old_pfn, numpages, psize, 2i32);
new_prot = verify_rwx(old_prot, new_prot, lpaddr, old_pfn, numpages, nx, rw);
// If there is a conflict, split the large page.
//
// There used to be a 4k wise evaluation trying really hard to
// preserve the large pages, but experimentation has shown, that this
// does not help at all. There might be corner cases which would
// preserve one large page occasionally, but it's really not worth the
// extra code and cycles for the common case.
if pgprot_val!(req_prot) != pgprot_val!(new_prot) {
return 1;
}
// All checks passed. Update the large page mapping.
new_pte = pfn_pte(old_pfn, new_prot);
__set_pmd_pte(kpte, address, new_pte);
unsafe {
(*cpa).flags = (*cpa).flags | 1;
}
cpa_inc_lp_preserved((level as u32) as i32);
0
}

fn should_split_large_page(kpte: *mut __SlateRecord_pgtable_64_types_h_21, address: u64, cpa: *mut cpa_data) -> i32 {
if ((unsafe { (*cpa).__slate_bits_0.__get_force_split() }) as i32) != 0 {
return 1;
}
let mut __UNIQUE_ID_guard_604: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(pgd_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_604) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_605: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(pgd_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_605) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__should_split_large_page(kpte, address, cpa)
}

fn split_set_pte(cpa: *mut cpa_data, pte: *mut __SlateRecord_pgtable_64_types_h_21, pfn: u64, mut ref_prot: pgprot, address: u64, size: u64) {
let mut __ret_once: bool = unsafe { core::mem::zeroed() };
let mut prot: core::mem::MaybeUninit<pgprot> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut pgprot = core::ptr::addr_of_mut!(prot) as *mut pgprot;
let mut npg: u32 = unsafe { core::mem::zeroed() };
unsafe {
npg = PFN_DOWN!(size) as u32;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// If should_split_large_page() discovered an inconsistent mapping,
// remove the invalid protection in the split mapping.
if ((*cpa).__slate_bits_0.__get_force_static_prot() as i32) != 0 {
// Hand in lpsize = 0 to enforce the protection mechanism
*__slate_slot_6 = static_protections(ref_prot, address, pfn, npg as u64, 0u64, 1i32);
if pgprot_val!(*__slate_slot_6) != pgprot_val!(ref_prot) {
// If this is splitting a PMD, fix it up. PUD splits cannot be
// fixed trivially as that would require to rescan the newly
// installed PMD mappings after returning from split_large_page()
// so an eventual further split can allocate the necessary PTE
// pages. Warn for now and revisit it in case this actually
// happens.
if size == _AC!() << 12i32 {
ref_prot = *__slate_slot_6;
} else {
let __ret_do_once: bool = true;
let __ret_cond: bool = __ret_do_once;
__ret_once = false;
if unlikely__2!(__ret_cond) != 0 && unlikely__2!(!__already_done) != 0 {
__already_done = true;
__ret_once = true;
}
let __v3: i64 = __ret_once as i64;
if __v3 != 0 {
while false {
}
_printk(c"\x014CPA: Cannot fixup static protections for PUD split\n".as_ptr() as *const u8);
}
}
}
}
set_pte!(pte, pfn_pte(pfn, ref_prot));
}
}

#[unsafe(link_section = ".data..once")]
static mut __already_done: bool = unsafe { core::mem::zeroed() };

fn __split_large_page(cpa: *mut cpa_data, kpte: *mut __SlateRecord_pgtable_64_types_h_21, address: u64, pbase: *mut __SlateRecord_pgtable_64_types_h_21) -> i32 {
let mut __slate_compound_2: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: pgprot = unsafe { core::mem::zeroed() };
let mut lpaddr: u64 = 0;
let mut lpinc: u64 = 0;
let mut ref_pfn: u64 = 0;
let mut pfn: u64 = 0;
let mut pfninc: u64 = 1;
let base: *mut page = virt_to_page!(pbase);
let mut i: u32 = 0;
let mut level: u32 = 0;
let mut ref_prot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ref_prot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut nx: bool = false;
let mut rw: bool = false;
let mut __UNIQUE_ID_guard_606: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(pgd_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_606) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_607: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(pgd_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_607) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
// Check for races, another CPU might have split this page
// up for us already:
let tmp: *mut __SlateRecord_pgtable_64_types_h_21 = _lookup_address_cpa(cpa, address, core::ptr::addr_of_mut!(level), core::ptr::addr_of_mut!(nx), core::ptr::addr_of_mut!(rw));
if tmp != kpte {
return 1;
}
paravirt_alloc_pte(unsafe { core::ptr::addr_of_mut!(init_mm) }, ((unsafe { base.offset_from(vmemmap_base as *mut page) }) as i64) as u64);
match level {
2 => {
ref_prot = unsafe { *{
__slate_compound_2 = pgprot { pgprot: pmd_flags(*(kpte as *mut __SlateRecord_pgtable_64_types_h_22)) };
core::ptr::addr_of_mut!(__slate_compound_2)
} };
// Clear PSE (aka _PAGE_PAT) and move
// PAT bit to correct position.
ref_prot = pgprot_large_2_4k(ref_prot);
ref_pfn = pmd_pfn(unsafe { *(kpte as *mut __SlateRecord_pgtable_64_types_h_22) });
lpaddr = address & !(_AC!() << 21i32).wrapping_sub(1);
lpinc = _AC!() << 12i32;
}
3 => {
ref_prot = unsafe { *{
__slate_compound_4 = pgprot { pgprot: pud_flags(*(kpte as *mut __SlateRecord_pgtable_types_h_368)) };
core::ptr::addr_of_mut!(__slate_compound_4)
} };
ref_pfn = pud_pfn(unsafe { *(kpte as *mut __SlateRecord_pgtable_types_h_368) });
pfninc = _AC!() << 21i32 >> 12i32;
lpaddr = address & !(_AC!() << 30i32).wrapping_sub(1);
lpinc = _AC!() << 21i32;
// Clear the PSE flags if the PRESENT flag is not set
// otherwise pmd_present() will return true even on a non
// present pmd.
if pgprot_val!(ref_prot) & _AT!(u64, 1i32) << 0i32 == 0 {
let __v5: u64 = ref_prot.pgprot;
let __v6: u64 = __v5 & !(_AT!(u64, 1i32) << 7i32);
ref_prot.pgprot = __v6;
}
}
_ => {
return 1;
}
}
ref_prot = pgprot_clear_protnone_bits(ref_prot);
// Get the target pfn from the original entry:
pfn = ref_pfn;
while i < 512 {
split_set_pte(cpa, pbase.wrapping_offset(i as isize), pfn, ref_prot, lpaddr, lpinc);
i = i.wrapping_add(1);
pfn = pfn.wrapping_add(pfninc);
lpaddr = lpaddr.wrapping_add(lpinc);
}
if virt_addr_valid!(address) {
let pfn: u64 = PFN_DOWN!(__pa!(address));
if unsafe { pfn_range_is_mapped(pfn, pfn.wrapping_add(1)) } {
split_page_count(level as i32);
}
}
// Install the new, split up pagetable.
//
// We use the standard kernel pagetable protections for the new
// pagetable protections, the actual ptes set above control the
// primary protection behavior:
__set_pmd_pte(kpte, address, mk_pte(base as *const page, unsafe { *{
__slate_compound_14 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64) };
core::ptr::addr_of_mut!(__slate_compound_14)
} }));
// Do a global flush tlb after splitting the large page
// and before we do the actual change page attribute in the PTE.
//
// Without this, we violate the TLB application note, that says:
// "The TLBs may contain both ordinary and large-page
//  translations for a 4-KByte range of linear addresses. This
//  may occur if software modifies the paging structures so that
//  the page size used for the address range changes. If the two
//  translations differ with respect to page frame or attributes
//  (e.g., permissions), processor behavior is undefined and may
//  be implementation-specific."
//
// We do this global tlb flush inside the cpa_lock, so that we
// don't allow any other cpu, with stale tlb entries change the
// page attribute in parallel, that also falls into the
// just split large page entry.
unsafe { flush_tlb_all() };
0
}

fn split_large_page(cpa: *mut cpa_data, kpte: *mut __SlateRecord_pgtable_64_types_h_21, address: u64) -> i32 {
spin_unlock(unsafe { core::ptr::addr_of_mut!(cpa_lock) });
if ((unsafe { (*cpa).__slate_bits_0.__get_init_mm_read_locked() }) as i32) != 0 {
mmap_read_unlock(unsafe { core::ptr::addr_of_mut!(init_mm) });
}
let pte: *mut __SlateRecord_pgtable_64_types_h_21 = alloc_hooks__4!(pte_alloc_one_kernel_noprof(unsafe { core::ptr::addr_of_mut!(init_mm) }));
if ((unsafe { (*cpa).__slate_bits_0.__get_init_mm_read_locked() }) as i32) != 0 {
mmap_read_lock(unsafe { core::ptr::addr_of_mut!(init_mm) });
}
spin_lock(unsafe { core::ptr::addr_of_mut!(cpa_lock) });
if pte.is_null() {
return 12i32.wrapping_neg();
}
if __split_large_page(cpa, kpte, address, pte) != 0 {
pte_free_kernel(unsafe { core::ptr::addr_of_mut!(init_mm) }, pte);
}
0
}

fn collapse_pmd_page(pmd: *mut __SlateRecord_pgtable_64_types_h_22, mut addr: u64, pgtables: *mut list_head) -> i32 {
let mut __slate_compound_9: pgprot = unsafe { core::mem::zeroed() };
let mut _pmd: __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(_pmd) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut old_pmd: __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(old_pmd) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut pte: *mut __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
let mut first: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(first) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut pfn: u64 = 0;
let mut pgprot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pgprot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// Check that the rest of PTEs are compatible with the first one
let mut i: i32 = 1;
let __v0: u64 = addr;
addr = __v0 & !(_AC!() << 21i32).wrapping_sub(1);
pte = pte_offset_kernel(pmd, addr);
first = unsafe { *pte };
pfn = pte_pfn(first);
// Make sure alignment is suitable
if PFN_PHYS!(pfn) & !(!(_AC!() << 21i32).wrapping_sub(1)) != 0 {
return 0;
}
// The page is 4k intentionally
if pte_flags(first) & _AT!(u64, 1i32) << 11i32 != 0 {
return 0;
}
let __v2: *mut __SlateRecord_pgtable_64_types_h_21 = pte;
pte = __v2.wrapping_offset(1 as isize);
while i < 512 {
let entry: __SlateRecord_pgtable_64_types_h_21 = unsafe { *pte };
if pte_present(entry) == 0 {
return 0;
}
if pte_flags(entry) != pte_flags(first) {
return 0;
}
if pte_pfn(entry) != pte_pfn(first).wrapping_add(i as u64) {
return 0;
}
i = i.wrapping_add(1);
let __v7: *mut __SlateRecord_pgtable_64_types_h_21 = pte;
pte = __v7.wrapping_offset(1 as isize);
}
old_pmd = unsafe { *pmd };
// Success: set up a large page
pgprot = pgprot_4k_2_large(unsafe { *{
__slate_compound_9 = pgprot { pgprot: pte_flags(first) };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
let __v10: u64 = pgprot.pgprot;
let __v11: u64 = __v10 | _AT!(u64, 1i32) << 7i32;
pgprot.pgprot = __v11;
_pmd = pfn_pmd(pfn, pgprot);
set_pmd!(pmd, _pmd);
// Queue the page table to be freed after TLB flush
list_add(unsafe { core::ptr::addr_of_mut!((*page_ptdesc!((vmemmap_base as *mut page).wrapping_offset(pmd_pfn(old_pmd) as isize))).__slate_anon_1.pt_list) }, pgtables);
let __v12: bool = virt_addr_valid!(addr) && unsafe { pfn_range_is_mapped(pfn, pfn.wrapping_add(1)) };
if __v12 {
collapse_page_count(2i32);
}
1
}

fn collapse_pud_page(pud: *mut __SlateRecord_pgtable_types_h_368, mut addr: u64, pgtables: *mut list_head) -> i32 {
let mut __slate_compound_10: pgprot = unsafe { core::mem::zeroed() };
let mut pfn: u64 = 0;
let mut pmd: *mut __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
let mut first: __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(first) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// To restore PUD page, all following PMDs must be compatible with the
// first one.
let mut i: i32 = 1;
if (unsafe { direct_gbpages }) == 0 {
return 0;
}
let __v0: u64 = addr;
addr = __v0 & !(_AC!() << 30i32).wrapping_sub(1);
pmd = pmd_offset(pud, addr);
first = unsafe { *pmd };
// To restore PUD page all PMD entries must be large and
// have suitable alignment
pfn = pmd_pfn(first);
if !pmd_leaf(first) || PFN_PHYS!(pfn) & !(!(_AC!() << 30i32).wrapping_sub(1)) != 0 {
return 0;
}
let __v2: *mut __SlateRecord_pgtable_64_types_h_22 = pmd;
pmd = __v2.wrapping_offset(1 as isize);
while i < 512 {
let entry: __SlateRecord_pgtable_64_types_h_22 = unsafe { *pmd };
let __v4: bool = pmd_present(entry) == 0 || !pmd_leaf(entry);
if __v4 {
return 0;
}
if pmd_flags(entry) != pmd_flags(first) {
return 0;
}
if pmd_pfn(entry) != pmd_pfn(first).wrapping_add(i.wrapping_mul(512) as u64) {
return 0;
}
i = i.wrapping_add(1);
let __v8: *mut __SlateRecord_pgtable_64_types_h_22 = pmd;
pmd = __v8.wrapping_offset(1 as isize);
}
// Restore PUD page and queue page table to be freed after TLB flush
list_add(unsafe { core::ptr::addr_of_mut!((*page_ptdesc!((vmemmap_base as *mut page).wrapping_offset(pud_pfn(*pud) as isize))).__slate_anon_1.pt_list) }, pgtables);
native_set_pud(pud, pfn_pud(pfn, unsafe { *{
__slate_compound_10 = pgprot { pgprot: pmd_flags(first) };
core::ptr::addr_of_mut!(__slate_compound_10)
} }));
let __v11: bool = virt_addr_valid!(addr) && unsafe { pfn_range_is_mapped(pfn, pfn.wrapping_add(1)) };
if __v11 {
collapse_page_count(3i32);
}
1
}

/// Collapse PMD and PUD pages in the kernel mapping around the address where
/// possible.
///
/// Caller must flush TLB and free page tables queued on the list before
/// touching the new entries. CPU must not see TLB entries of different size
/// with different attributes.
fn collapse_large_pages(mut addr: u64, pgtables: *mut list_head) -> i32 {
let mut collapsed: i32 = 0;
let __v0: u64 = addr;
addr = __v0 & !(_AC!() << 21i32).wrapping_sub(1);
let mut __UNIQUE_ID_guard_610: __SlateRecord_spinlock_h_610 = class_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(pgd_lock) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_610) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_spinlock_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_611: *mut __SlateRecord_spinlock_h_610 = ((unsafe { core::ptr::addr_of_mut!(pgd_lock) }) as u64) as *mut __SlateRecord_spinlock_h_610;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_611) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
let pgd: *mut __SlateRecord_pgtable_types_h_295 = pgd_offset_k!(addr);
if pgd_none(unsafe { *pgd }) != 0 {
return 0;
}
let p4d: *mut __SlateRecord_pgtable_types_h_342 = p4d_offset(pgd, addr);
if p4d_none(unsafe { *p4d }) != 0 {
return 0;
}
let pud: *mut __SlateRecord_pgtable_types_h_368 = pud_offset(p4d, addr);
let __v4: bool = pud_present(unsafe { *pud }) == 0 || pud_leaf(unsafe { *pud });
if __v4 {
return 0;
}
let pmd: *mut __SlateRecord_pgtable_64_types_h_22 = pmd_offset(pud, addr);
let __v5: bool = pmd_present(unsafe { *pmd }) == 0 || pmd_leaf(unsafe { *pmd });
if __v5 {
return 0;
}
collapsed = collapse_pmd_page(pmd, addr, pgtables);
if collapsed != 0 {
let __v6: i32 = collapse_pud_page(pud, addr, pgtables);
collapsed = collapsed.wrapping_add(__v6);
}
collapsed
}

fn try_to_free_pte_page(pte: *mut __SlateRecord_pgtable_64_types_h_21) -> bool {
for i in 0i32..512i32 {
if pte_none(unsafe { *pte.wrapping_offset(i as isize) }) == 0 {
return false;
}
}
pte_free_kernel(unsafe { core::ptr::addr_of_mut!(init_mm) }, pte);
true
}

fn try_to_free_pmd_page(pmd: *mut __SlateRecord_pgtable_64_types_h_22) -> bool {
for i in 0i32..512i32 {
if pmd_none(unsafe { *pmd.wrapping_offset(i as isize) }) == 0 {
return false;
}
}
pmd_free(unsafe { core::ptr::addr_of_mut!(init_mm) }, pmd);
true
}

fn unmap_pte_range(pmd: *mut __SlateRecord_pgtable_64_types_h_22, mut start: u64, end: u64) -> bool {
let mut pte: *mut __SlateRecord_pgtable_64_types_h_21 = pte_offset_kernel(pmd, start);
while start < end {
set_pte!(pte, __pte__2!(0i32));
let __v0: u64 = start;
start = __v0.wrapping_add(_AC!() << 12i32);
let __v2: *mut __SlateRecord_pgtable_64_types_h_21 = pte;
pte = __v2.wrapping_offset(1 as isize);
}
if try_to_free_pte_page(pmd_page_vaddr(unsafe { *pmd }) as *mut __SlateRecord_pgtable_64_types_h_21) {
pmd_clear!(pmd);
return true;
}
false
}

fn __unmap_pmd_range(pud: *mut __SlateRecord_pgtable_types_h_368, pmd: *mut __SlateRecord_pgtable_64_types_h_22, start: u64, end: u64) {
if unmap_pte_range(pmd, start, end) {
if try_to_free_pmd_page(pud_pgtable(unsafe { *pud })) {
pud_clear!(pud);
}
}
}

fn unmap_pmd_range(pud: *mut __SlateRecord_pgtable_types_h_368, mut start: u64, end: u64) {
let mut pmd: *mut __SlateRecord_pgtable_64_types_h_22 = pmd_offset(pud, start);
// Not on a 2MB page boundary?
if start & (_AC!() << 21i32).wrapping_sub(1) != 0 {
let next_page: u64 = start.wrapping_add(_AC!() << 21i32) & !(_AC!() << 21i32).wrapping_sub(1);
let pre_end: u64 = min_t!(u64, end, next_page);
__unmap_pmd_range(pud, pmd, start, pre_end);
start = pre_end;
let __v3: *mut __SlateRecord_pgtable_64_types_h_22 = pmd;
pmd = __v3.wrapping_offset(1 as isize);
}
// Try to unmap in 2M chunks.
while end.wrapping_sub(start) >= _AC!() << 21i32 {
if pmd_leaf(unsafe { *pmd }) {
pmd_clear!(pmd);
} else {
__unmap_pmd_range(pud, pmd, start, start.wrapping_add(_AC!() << 21i32));
}
let __v5: u64 = start;
start = __v5.wrapping_add(_AC!() << 21i32);
let __v7: *mut __SlateRecord_pgtable_64_types_h_22 = pmd;
pmd = __v7.wrapping_offset(1 as isize);
}
// 4K leftovers?
if start < end {
__unmap_pmd_range(pud, pmd, start, end);
return;
}
// Try again to free the PMD page if haven't succeeded above.
if pud_none(unsafe { *pud }) == 0 {
if try_to_free_pmd_page(pud_pgtable(unsafe { *pud })) {
pud_clear!(pud);
}
}
}

fn unmap_pud_range(p4d: *mut __SlateRecord_pgtable_types_h_342, mut start: u64, end: u64) {
let mut pud: *mut __SlateRecord_pgtable_types_h_368 = pud_offset(p4d, start);
// Not on a GB page boundary?
if start & (_AC!() << 30i32).wrapping_sub(1) != 0 {
let next_page: u64 = start.wrapping_add(_AC!() << 30i32) & !(_AC!() << 30i32).wrapping_sub(1);
let pre_end: u64 = min_t!(u64, end, next_page);
unmap_pmd_range(pud, start, pre_end);
start = pre_end;
let __v3: *mut __SlateRecord_pgtable_types_h_368 = pud;
pud = __v3.wrapping_offset(1 as isize);
}
// Try to unmap in 1G chunks?
while end.wrapping_sub(start) >= _AC!() << 30i32 {
if pud_leaf(unsafe { *pud }) {
pud_clear!(pud);
} else {
unmap_pmd_range(pud, start, start.wrapping_add(_AC!() << 30i32));
}
let __v5: u64 = start;
start = __v5.wrapping_add(_AC!() << 30i32);
let __v7: *mut __SlateRecord_pgtable_types_h_368 = pud;
pud = __v7.wrapping_offset(1 as isize);
}
// 2M leftovers?
if start < end {
unmap_pmd_range(pud, start, end);
}
// No need to try to free the PUD page because we'll free it in
// populate_pgd's error path
}

fn alloc_pte_page(pmd: *mut __SlateRecord_pgtable_64_types_h_22) -> i32 {
let pte: *mut __SlateRecord_pgtable_64_types_h_21 = alloc_hooks__4!(pte_alloc_one_kernel_noprof(unsafe { core::ptr::addr_of_mut!(init_mm) }));
if pte.is_null() {
return 1i32.wrapping_neg();
}
set_pmd!(pmd, __pmd!(__pa!(pte) | (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64))));
0
}

fn alloc_pmd_page(pud: *mut __SlateRecord_pgtable_types_h_368) -> i32 {
// Pass 0 as a placeholder for the second argument, since the
// generic implementation of pmd_alloc_one() does not use it.
let pmd: *mut __SlateRecord_pgtable_64_types_h_22 = alloc_hooks__5!(pmd_alloc_one_noprof(unsafe { core::ptr::addr_of_mut!(init_mm) }, 0u64));
if pmd.is_null() {
return 1i32.wrapping_neg();
}
set_pud!(pud, __pud!(__pa!(pmd) | (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64))));
0
}

fn populate_pte(cpa: *mut cpa_data, mut start: u64, end: u64, mut num_pages: u32, pmd: *mut __SlateRecord_pgtable_64_types_h_22, mut pgprot: pgprot) {
let mut pte: *mut __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
pte = pte_offset_kernel(pmd, start);
pgprot = pgprot_clear_protnone_bits(pgprot);
loop {
let __v7: u32 = num_pages;
num_pages = __v7.wrapping_sub(1);
if !(__v7 != 0 && start < end) {
break;
}
set_pte!(pte, pfn_pte(unsafe { (*cpa).pfn }, pgprot));
let __v0: u64 = start;
start = __v0.wrapping_add(_AC!() << 12i32);
let __v3: u64 = unsafe { (*cpa).pfn };
unsafe {
(*cpa).pfn = __v3.wrapping_add(1);
}
let __v5: *mut __SlateRecord_pgtable_64_types_h_21 = pte;
pte = __v5.wrapping_offset(1 as isize);
}
}

fn populate_pmd(cpa: *mut cpa_data, mut start: u64, end: u64, num_pages: u32, pud: *mut __SlateRecord_pgtable_types_h_368, pgprot: pgprot) -> i64 {
let mut __slate_compound_6: pgprot = unsafe { core::mem::zeroed() };
let mut cur_pages: i64 = 0;
let mut pmd: *mut __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
let mut pmd_pgprot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pmd_pgprot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// Not on a 2M boundary?
if start & (_AC!() << 21i32).wrapping_sub(1) != 0 {
let mut pre_end: u64 = start.wrapping_add((num_pages << 12i32) as u64);
let next_page: u64 = start.wrapping_add(_AC!() << 21i32) & !(_AC!() << 21i32).wrapping_sub(1);
pre_end = min_t!(u64, pre_end, next_page);
cur_pages = (pre_end.wrapping_sub(start) >> 12i32) as i64;
cur_pages = min_t__3!(u32, num_pages, cur_pages) as i64;
// Need a PTE page?
pmd = pmd_offset(pud, start);
if pmd_none(unsafe { *pmd }) != 0 {
if alloc_pte_page(pmd) != 0 {
return 1i32.wrapping_neg() as i64;
}
}
populate_pte(cpa, start, pre_end, cur_pages as u32, pmd, pgprot);
start = pre_end;
}
// We mapped them all?
if (num_pages as i64) == cur_pages {
return cur_pages;
}
pmd_pgprot = pgprot_4k_2_large(pgprot);
'__slate_break_14: while end.wrapping_sub(start) >= _AC!() << 21i32 {
// We cannot use a 1G page so allocate a PMD page if needed.
if pud_none(unsafe { *pud }) != 0 {
if alloc_pmd_page(pud) != 0 {
return 1i32.wrapping_neg() as i64;
}
}
pmd = pmd_offset(pud, start);
native_set_pmd(pmd, pmd_mkhuge(pfn_pmd(unsafe { (*cpa).pfn }, unsafe { *{
__slate_compound_6 = pgprot { pgprot: massage_pgprot(pmd_pgprot) };
core::ptr::addr_of_mut!(__slate_compound_6)
} })));
let __v7: u64 = start;
start = __v7.wrapping_add(_AC!() << 21i32);
let __v10: u64 = unsafe { (*cpa).pfn };
let __v11: u64 = __v10.wrapping_add(_AC!() << 21i32 >> 12i32);
unsafe {
(*cpa).pfn = __v11;
}
let __v12: i64 = cur_pages;
cur_pages = (__v12 as u64).wrapping_add(_AC!() << 21i32 >> 12i32) as i64;
}
// Map trailing 4K pages.
if start < end {
pmd = pmd_offset(pud, start);
if pmd_none(unsafe { *pmd }) != 0 {
if alloc_pte_page(pmd) != 0 {
return 1i32.wrapping_neg() as i64;
}
}
populate_pte(cpa, start, end, (num_pages as i64).wrapping_sub(cur_pages) as u32, pmd, pgprot);
}
num_pages as i64
}

fn populate_pud(cpa: *mut cpa_data, mut start: u64, p4d: *mut __SlateRecord_pgtable_types_h_342, pgprot: pgprot) -> i32 {
let mut __slate_compound_6: pgprot = unsafe { core::mem::zeroed() };
let mut pud: *mut __SlateRecord_pgtable_types_h_368 = unsafe { core::mem::zeroed() };
let mut end: u64 = 0;
let mut cur_pages: i64 = 0;
let mut pud_pgprot: pgprot = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pud_pgprot) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
end = start.wrapping_add((unsafe { (*cpa).numpages }) << 12i32);
// Not on a Gb page boundary? => map everything up to it with
// smaller pages.
if start & (_AC!() << 30i32).wrapping_sub(1) != 0 {
let mut pre_end: u64 = 0;
let next_page: u64 = start.wrapping_add(_AC!() << 30i32) & !(_AC!() << 30i32).wrapping_sub(1);
pre_end = min_t!(u64, end, next_page);
cur_pages = (pre_end.wrapping_sub(start) >> 12i32) as i64;
cur_pages = min_t__3!(i32, (unsafe { (*cpa).numpages }) as i32, cur_pages) as i64;
pud = pud_offset(p4d, start);
// Need a PMD page?
if pud_none(unsafe { *pud }) != 0 {
if alloc_pmd_page(pud) != 0 {
return 1i32.wrapping_neg();
}
}
cur_pages = populate_pmd(cpa, start, pre_end, cur_pages as u32, pud, pgprot);
if cur_pages < 0 {
return cur_pages as i32;
}
start = pre_end;
}
// We mapped them all?
if (unsafe { (*cpa).numpages }) == (cur_pages as u64) {
return cur_pages as i32;
}
pud = pud_offset(p4d, start);
pud_pgprot = pgprot_4k_2_large(pgprot);
// Map everything starting from the Gb boundary, possibly with 1G pages
'__slate_break_17: while boot_cpu_has!(58i32) != 0 && end.wrapping_sub(start) >= _AC!() << 30i32 {
native_set_pud(pud, pud_mkhuge(pfn_pud(unsafe { (*cpa).pfn }, unsafe { *{
__slate_compound_6 = pgprot { pgprot: massage_pgprot(pud_pgprot) };
core::ptr::addr_of_mut!(__slate_compound_6)
} })));
let __v7: u64 = start;
start = __v7.wrapping_add(_AC!() << 30i32);
let __v10: u64 = unsafe { (*cpa).pfn };
let __v11: u64 = __v10.wrapping_add(_AC!() << 30i32 >> 12i32);
unsafe {
(*cpa).pfn = __v11;
}
let __v12: i64 = cur_pages;
cur_pages = (__v12 as u64).wrapping_add(_AC!() << 30i32 >> 12i32) as i64;
let __v14: *mut __SlateRecord_pgtable_types_h_368 = pud;
pud = __v14.wrapping_offset(1 as isize);
}
// Map trailing leftover
if start < end {
let mut tmp: i64 = 0;
pud = pud_offset(p4d, start);
if pud_none(unsafe { *pud }) != 0 {
if alloc_pmd_page(pud) != 0 {
return 1i32.wrapping_neg();
}
}
tmp = populate_pmd(cpa, start, end, unsafe { (*cpa).numpages }.wrapping_sub(cur_pages as u64) as u32, pud, pgprot);
if tmp < 0 {
return cur_pages as i32;
}
cur_pages = cur_pages.wrapping_add(tmp);
}
cur_pages as i32
}

/// Restrictions for kernel page table do not necessarily apply when mapping in
/// an alternate PGD.
fn populate_pgd(cpa: *mut cpa_data, addr: u64) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut pgprot: pgprot = unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
let mut pud: *mut __SlateRecord_pgtable_types_h_368 = core::ptr::null_mut::<__SlateRecord_pgtable_types_h_368>(); // shut up gcc
let mut p4d: *mut __SlateRecord_pgtable_types_h_342 = unsafe { core::mem::zeroed() };
let mut ret: i64 = 0;
let pgd_entry: *mut __SlateRecord_pgtable_types_h_295 = unsafe { (*cpa).pgd }.wrapping_offset(pgd_index!(addr) as isize);
if pgd_none(unsafe { *pgd_entry }) != 0 {
// Pass 0 as a placeholder for the second argument, since the
// generic implementation of p4d_alloc_one() does not use it.
p4d = alloc_hooks__8!(p4d_alloc_one_noprof(unsafe { core::ptr::addr_of_mut!(init_mm) }, 0u64));
if p4d.is_null() {
return 1i32.wrapping_neg();
}
set_pgd!(pgd_entry, __pgd!(__pa!(p4d) | (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64))));
}
// Allocate a PUD page and hand it down for mapping.
p4d = p4d_offset(pgd_entry, addr);
if p4d_none(unsafe { *p4d }) != 0 {
// Pass 0 as a placeholder for the second argument, since the
// generic implementation of pud_alloc_one() does not use it.
pud = alloc_hooks__6!(pud_alloc_one_noprof(unsafe { core::ptr::addr_of_mut!(init_mm) }, 0u64));
if pud.is_null() {
return 1i32.wrapping_neg();
}
set_p4d!(p4d, __p4d!(__pa!(pud) | (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64))));
}
let __v11: u64 = pgprot.pgprot;
let __v12: u64 = __v11 & !pgprot_val__2!((*cpa).mask_clr);
pgprot.pgprot = __v12;
let __v13: u64 = pgprot.pgprot;
let __v14: u64 = __v13 | pgprot_val__2!((*cpa).mask_set);
pgprot.pgprot = __v14;
ret = populate_pud(cpa, addr, p4d, pgprot) as i64;
if ret < 0 {
// Leave the PUD page in place in case some other CPU or thread
// already found it, but remove any useless entries we just
// added to it.
unmap_pud_range(p4d, addr, addr.wrapping_add((unsafe { (*cpa).numpages }) << 12i32));
return ret as i32;
}
unsafe {
(*cpa).numpages = ret as u64;
}
0
}

fn __cpa_process_fault(cpa: *mut cpa_data, vaddr: u64, primary: i32) -> i32 {
if !unsafe { (*cpa).pgd }.is_null() {
// Right now, we only execute this code path when mapping
// the EFI virtual memory map regions, no other users
// provide a ->pgd value. This may change in the future.
return populate_pgd(cpa, vaddr);
}
// Ignore all non primary paths.
if primary == 0 {
unsafe {
(*cpa).numpages = 1;
}
return 0;
}
// Ignore the NULL PTE for kernel identity mapping, as it is expected
// to have holes.
// Also set numpages to '1' indicating that we processed cpa req for
// one virtual address page and its pfn. TBD: numpages can be set based
// on the initial value and the level returned by lookup_address().
if within(vaddr, unsafe { page_offset_base }, unsafe { page_offset_base }.wrapping_add((unsafe { max_pfn_mapped }) << 12i32)) != 0 {
unsafe {
(*cpa).numpages = 1;
}
unsafe {
(*cpa).pfn = __pa!(vaddr) >> 12i32;
}
return 0;
} else {
if __cpa_pfn_in_highmap(unsafe { (*cpa).pfn }) {
// Faults in the highmap are OK, so do not warn:
return 14i32.wrapping_neg();
} else {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"\x014CPA: called for zero pte. vaddr = %lx cpa->vaddr = %lx\n".as_ptr() as *const u8, vaddr, *(*cpa).vaddr) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string___cpa_process_fault_3, sym __slate_asm_string___cpa_process_fault_4, const 1854 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, vaddr, *(*cpa).vaddr) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return 14i32.wrapping_neg();
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_624: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __change_page_attr(cpa: *mut cpa_data, primary: i32) -> i32 {
let mut __slate_compound_19: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_20: pgprot = unsafe { core::mem::zeroed() };
let mut new_prot: core::mem::MaybeUninit<pgprot> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut pgprot = core::ptr::addr_of_mut!(new_prot) as *mut pgprot;
let mut old_prot: core::mem::MaybeUninit<pgprot> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut pgprot = core::ptr::addr_of_mut!(old_prot) as *mut pgprot;
let mut new_pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(new_pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut rw: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut bool = core::ptr::addr_of_mut!(rw) as *mut bool;
let mut nx: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut bool = core::ptr::addr_of_mut!(nx) as *mut bool;
let mut old_pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(old_pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut kpte: core::mem::MaybeUninit<*mut __SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(kpte) as *mut *mut __SlateRecord_pgtable_64_types_h_21;
let mut level: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(level) as *mut u32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut do_split: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(do_split) as *mut i32;
let mut address: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u64 = core::ptr::addr_of_mut!(address) as *mut u64;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_8 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 1usize);
*__slate_slot_18 = __cpa_addr(cpa, (*cpa).curpage);
'__join_6: {
loop {
*__slate_slot_14 = _lookup_address_cpa(cpa, *__slate_slot_18, core::ptr::addr_of_mut!(*__slate_slot_15), core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11));
if (*__slate_slot_14).is_null() {
return __cpa_process_fault(cpa, *__slate_slot_18, primary);
} else {
*__slate_slot_13 = *(*__slate_slot_14);
if pte_none(*__slate_slot_13) != 0 {
return __cpa_process_fault(cpa, *__slate_slot_18, primary);
} else {
if *__slate_slot_15 == 1 {
break '__join_6;
} else {
// Check, whether we can keep the large page intact
// and just change the pte:
*__slate_slot_17 = should_split_large_page(*__slate_slot_14, *__slate_slot_18, cpa);
// When the range fits into the existing large page,
// return. cp->numpages and cpa->tlbflush have been updated in
// try_large_page:
if *__slate_slot_17 <= 0 {
return *__slate_slot_17;
} else {
// We have to split the large page:
*__slate_slot_16 = split_large_page(cpa, *__slate_slot_14, *__slate_slot_18);
if *__slate_slot_16 != 0 {
break;
}
}
}
}
}
}
return *__slate_slot_16;
}
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_9, *{
__slate_compound_19 = pgprot { pgprot: pte_flags(*__slate_slot_13) };
core::ptr::addr_of_mut!(__slate_compound_19)
});
core::ptr::write(__slate_slot_8, *{
__slate_compound_20 = pgprot { pgprot: pte_flags(*__slate_slot_13) };
core::ptr::addr_of_mut!(__slate_compound_20)
});
let pfn: u64 = pte_pfn(*__slate_slot_13);
let __v6: u64 = (*__slate_slot_8).pgprot;
let __v5: u64 = __v6 & !pgprot_val__2!((*cpa).mask_clr);
(*__slate_slot_8).pgprot = __v5;
let __v4: u64 = (*__slate_slot_8).pgprot;
let __v3: u64 = __v4 | pgprot_val__2!((*cpa).mask_set);
(*__slate_slot_8).pgprot = __v3;
cpa_inc_4k_install();
// Hand in lpsize = 0 to enforce the protection mechanism
*__slate_slot_8 = static_protections(*__slate_slot_8, *__slate_slot_18, pfn, 1u64, 0u64, 1i32);
*__slate_slot_8 = verify_rwx(*__slate_slot_9, *__slate_slot_8, *__slate_slot_18, pfn, 1u64, *__slate_slot_12, *__slate_slot_11);
*__slate_slot_8 = pgprot_clear_protnone_bits(*__slate_slot_8);
// We need to keep the pfn from the existing PTE,
// after all we're only going to change its attributes
// not the memory it points to
*__slate_slot_10 = pfn_pte(pfn, *__slate_slot_8);
(*cpa).pfn = pfn;
// Do we really change anything ?
if pte_val!(*__slate_slot_13) != pte_val!(*__slate_slot_10) {
set_pte_atomic!(*__slate_slot_14, *__slate_slot_10);
(*cpa).flags = (*cpa).flags | 1;
}
(*cpa).numpages = 1;
return 0;
}
}

/// Check the directmap and "high kernel map" 'aliases'.
fn cpa_process_alias(cpa: *mut cpa_data) -> i32 {
let mut alias_cpa: cpa_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(alias_cpa) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let mut laddr: u64 = __va!((unsafe { (*cpa).pfn }) << 12i32) as u64;
let mut vaddr: u64 = 0;
let mut ret: i32 = 0;
if !unsafe { pfn_range_is_mapped((*cpa).pfn, unsafe { (*cpa).pfn }.wrapping_add(1)) } {
return 0;
}
// No need to redo, when the primary call touched the direct
// mapping already:
vaddr = __cpa_addr(cpa, unsafe { (*cpa).curpage });
if within(vaddr, unsafe { page_offset_base }, unsafe { page_offset_base }.wrapping_add((unsafe { max_pfn_mapped }) << 12i32)) == 0 {
alias_cpa = unsafe { *cpa };
alias_cpa.vaddr = core::ptr::addr_of_mut!(laddr);
let __v1: u32 = alias_cpa.flags & 4294967289u32;
alias_cpa.flags = __v1;
alias_cpa.curpage = 0;
// Directmap always has NX set, do not modify.
if (unsafe { __supported_pte_mask }) & _AT!(u64, 1i32) << 63i32 != 0 {
let __v2: u64 = alias_cpa.mask_clr.pgprot;
let __v3: u64 = __v2 & !(_AT!(u64, 1i32) << 63i32);
alias_cpa.mask_clr.pgprot = __v3;
let __v4: u64 = alias_cpa.mask_set.pgprot;
let __v5: u64 = __v4 & !(_AT!(u64, 1i32) << 63i32);
alias_cpa.mask_set.pgprot = __v5;
}
unsafe {
(*cpa).__slate_bits_0.__set_force_flush_all(1u32);
}
ret = __change_page_attr_set_clr(core::ptr::addr_of_mut!(alias_cpa), 0i32);
if ret != 0 {
return ret;
}
}
// If the primary call didn't touch the high mapping already
// and the physical address is inside the kernel map, we need
// to touch the high mapped kernel as well:
let __v6: bool = within(vaddr, (unsafe { core::ptr::addr_of_mut!(_text) as *mut u8 }) as u64, unsafe { _brk_end }) == 0 && __cpa_pfn_in_highmap(unsafe { (*cpa).pfn });
if __v6 {
let mut temp_cpa_vaddr: u64 = ((unsafe { (*cpa).pfn }) << 12i32).wrapping_add(_AC__2!()).wrapping_sub(unsafe { phys_base });
alias_cpa = unsafe { *cpa };
alias_cpa.vaddr = core::ptr::addr_of_mut!(temp_cpa_vaddr);
let __v8: u32 = alias_cpa.flags & 4294967289u32;
alias_cpa.flags = __v8;
alias_cpa.curpage = 0;
// [_text, _brk_end) also covers data, do not modify NX except
// in cases where the highmap is the primary target.
if (unsafe { __supported_pte_mask }) & _AT!(u64, 1i32) << 63i32 != 0 {
let __v9: u64 = alias_cpa.mask_clr.pgprot;
let __v10: u64 = __v9 & !(_AT!(u64, 1i32) << 63i32);
alias_cpa.mask_clr.pgprot = __v10;
let __v11: u64 = alias_cpa.mask_set.pgprot;
let __v12: u64 = __v11 & !(_AT!(u64, 1i32) << 63i32);
alias_cpa.mask_set.pgprot = __v12;
}
unsafe {
(*cpa).__slate_bits_0.__set_force_flush_all(1u32);
}
// The high mapping range is imprecise, so ignore the
// return value.
__change_page_attr_set_clr(core::ptr::addr_of_mut!(alias_cpa), 0i32);
}
0
}

fn __change_page_attr_set_clr(cpa: *mut cpa_data, primary: i32) -> i32 {
let mut __UNIQUE_ID_unlock_626: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_626) as *mut *mut __SlateRecord_spinlock_h_610;
let mut __UNIQUE_ID_guard_625: core::mem::MaybeUninit<__SlateRecord_spinlock_h_610> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut __SlateRecord_spinlock_h_610 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_625) as *mut __SlateRecord_spinlock_h_610;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut lock: bool = unsafe { core::mem::zeroed() };
let mut rempages: u64 = unsafe { core::mem::zeroed() };
let mut numpages: u64 = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_6 = __slate_slot_6;
let __slate_cleanup_armed_6 = core::cell::Cell::new(false);
let __slate_cleanup_6 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_6.get() {
unsafe { class_spinlock_destructor(__slate_cleanup_ptr_6 as *mut _) };
}
});
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { __class_spinlock_cleanup_ctx(__slate_cleanup_ptr_5 as *mut _) };
}
});
unsafe {
numpages = (*cpa).numpages;
rempages = numpages;
lock = true;
ret = 0;
// No changes, easy!
if pgprot_val__2!((*cpa).mask_set) | pgprot_val__2!((*cpa).mask_clr) == 0 && ((*cpa).__slate_bits_0.__get_force_split() as i32) == 0 {
return ret;
} else {
// DEBUG_PAGEALLOC is special; it is called from any context the
// page-allocator is, which violates the normal cpa_lock locking
// rules.
//
// However, since it is part of the page-allocator, things are still
// properly serialized by the page-allocator locking and the fact that
// when a page is owned by the page-allocator, it isn't owned by
// anybody else. That is, you *SHOULD NOT* be calling cpa() on memory
// that isn't allocated.
//
// Additionally, DEBUG_PAGEALLOC ensures (per probe_page_size_mask())
// that the kernel mapping is 4k pages, therefore there are no large
// pages to split/collapse.
//
// Furthermore, the page-allocator strictly manages pages that
// *exist*, avoiding pgd_lock.
//
// Therefore, it is safe to not take cpa_lock.
if debug_pagealloc_enabled() && (*cpa).flags & 32 != 0 {
lock = false;
}
'__join_0: {
'__loop_1: loop {
if rempages != 0 {
// Store the remaining nr of pages for the large page
// preservation check.
(*cpa).numpages = rempages;
// for array changes, we can't use large page
if (*cpa).flags & 6u32 != 0 {
(*cpa).numpages = 1;
}
if lock {
core::ptr::write(__slate_slot_6, class_spinlock_constructor(core::ptr::addr_of_mut!(cpa_lock)));
__slate_cleanup_armed_6.set(true);
core::ptr::write(__slate_slot_5, (core::ptr::addr_of_mut!(cpa_lock) as u64) as *mut __SlateRecord_spinlock_h_610);
__slate_cleanup_armed_5.set(true);
ret = __change_page_attr(cpa, primary);
__slate_cleanup_armed_5.set(false);
__class_spinlock_cleanup_ctx(__slate_slot_5 as *mut _);
__slate_cleanup_armed_6.set(false);
class_spinlock_destructor(__slate_slot_6 as *mut _);
} else {
ret = __change_page_attr(cpa, primary);
}
if ret != 0 {
break '__join_0;
} else {
if primary != 0 && (*cpa).flags & 8 == 0 {
ret = cpa_process_alias(cpa);
if ret != 0 {
break '__join_0;
}
}
// Adjust the number of pages with the result of the
// CPA operation. Either a large page has been
// preserved or a single page update happened.
loop {
if unlikely__2!((*cpa).numpages > rempages || (*cpa).numpages == 0) != 0 {
break '__loop_1;
} else {
if true {
break;
}
}
}
rempages = rempages.wrapping_sub((*cpa).numpages);
let __v1: u64 = (*cpa).curpage;
(*cpa).curpage = __v1.wrapping_add((*cpa).numpages);
}
} else {
break '__join_0;
}
}
core::arch::asm!("627: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 627b - ., 3; .popsection\n# {0}", const 627 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___change_page_attr_set_clr_11, sym __slate_asm_string___change_page_attr_set_clr_12, const 2082 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
(*cpa).numpages = numpages;
// Restore the original numpages
return ret;
}
}
}

fn change_page_attr_set_clr(addr: *mut u64, numpages: i32, mut mask_set: pgprot, mask_clr: pgprot, force_split: i32, in_flag: i32, pages: *mut *mut page) -> i32 {
let mut __slate_compound_23: pgprot = unsafe { core::mem::zeroed() };
let mut scope: core::mem::MaybeUninit<*mut mm_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut mm_struct = core::ptr::addr_of_mut!(scope) as *mut *mut mm_struct;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut cache: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(cache) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut cpa: core::mem::MaybeUninit<cpa_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut cpa_data = core::ptr::addr_of_mut!(cpa) as *mut cpa_data;
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_mmap_read_lock_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
memset(core::ptr::addr_of_mut!(*__slate_slot_18) as *mut (), 0i32, core::mem::size_of::<cpa_data>() as u64);
// Check, if we are requested to set a not supported
// feature.  Clearing non-supported features is OK.
mask_set = *{
__slate_compound_23 = pgprot { pgprot: massage_pgprot(mask_set) };
core::ptr::addr_of_mut!(__slate_compound_23)
};
if pgprot_val!(mask_set) == 0 && pgprot_val!(mask_clr) == 0 && force_split == 0 {
return 0;
} else {
// Ensure we are PAGE_SIZE aligned
if in_flag & 2 != 0 {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_9 = 0;
while *__slate_slot_9 < numpages {
if *addr.wrapping_offset(*__slate_slot_9 as isize) & !(!(_AC!() << 12i32).wrapping_sub(1)) != 0 {
let __v8: *mut u64 = addr.wrapping_offset(*__slate_slot_9 as isize);
let __v7: u64 = *__v8;
let __v6: u64 = __v7 & !(_AC!() << 12i32).wrapping_sub(1);
*__v8 = __v6;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("628: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 628b - ., 3; .popsection\n# {0}", const 628 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_change_page_attr_set_clr_19, sym __slate_asm_string_change_page_attr_set_clr_20, const 2118 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("629: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 629b - ., 4; .popsection\n# {0}", const 629 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
let __v2: i32 = *__slate_slot_9;
*__slate_slot_9 = __v2.wrapping_add(1);
}
} else {
if in_flag & 4 == 0 {
// in_flag of CPA_PAGES_ARRAY implies it is aligned.
// No need to check in that case
if *addr & !(!(_AC!() << 12i32).wrapping_sub(1)) != 0 {
let __v14: u64 = *addr;
let __v13: u64 = __v14 & !(_AC!() << 12i32).wrapping_sub(1);
*addr = __v13;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("630: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 630b - ., 3; .popsection\n# {0}", const 630 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_change_page_attr_set_clr_21, sym __slate_asm_string_change_page_attr_set_clr_22, const 2131 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("631: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 631b - ., 4; .popsection\n# {0}", const 631 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// People should not be passing in unaligned addresses:
}
}
}
// Must avoid aliasing mappings in the highmem code
kmap_flush_unused();
vm_unmap_aliases();
(*__slate_slot_18).vaddr = addr;
(*__slate_slot_18).pages = pages;
(*__slate_slot_18).numpages = numpages as u64;
(*__slate_slot_18).mask_set = mask_set;
(*__slate_slot_18).mask_clr = mask_clr;
(*__slate_slot_18).flags = in_flag as u32;
(*__slate_slot_18).curpage = 0;
(*__slate_slot_18).__slate_bits_0.__set_force_split(force_split as u32);
// Avoid race with concurrent CPA collapse.
(*__slate_slot_18).__slate_bits_0.__set_init_mm_read_locked(1u32);
core::ptr::write(__slate_slot_0, class_mmap_read_lock_constructor__const9(core::ptr::addr_of_mut!(init_mm)));
__slate_cleanup_armed_0.set(true);
if !class_mmap_read_lock_lock_ptr(&(*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_mmap_read_lock_is_conditional {
if true {
*__slate_slot_17 = __change_page_attr_set_clr(core::ptr::addr_of_mut!(*__slate_slot_18), 1i32);
}
}
__slate_cleanup_armed_0.set(false);
class_mmap_read_lock_destructor(__slate_slot_0 as *mut _);
(*__slate_slot_18).__slate_bits_0.__set_init_mm_read_locked(0u32);
// Check whether we really changed something:
if (*__slate_slot_18).flags & 1 != 0 {
// No need to flush, when we did not set any of the caching
// attributes:
*__slate_slot_16 = ((pgprot2cachemode(mask_set) as u32) != 0) as i32;
// On error; flush everything to be sure.
if *__slate_slot_17 != 0 {
cpa_flush_all(*__slate_slot_16 as u64);
} else {
cpa_flush(core::ptr::addr_of_mut!(*__slate_slot_18), *__slate_slot_16);
}
}
return *__slate_slot_17;
}
}
}

fn change_page_attr_set(addr: *mut u64, numpages: i32, mask: pgprot, array: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set_clr(addr, numpages, mask, unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32, if array != 0 { 2i32 } else { 0i32 }, core::ptr::null_mut::<*mut page>())
}

fn change_page_attr_clear(addr: *mut u64, numpages: i32, mask: pgprot, array: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set_clr(addr, numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, mask, 0i32, if array != 0 { 2i32 } else { 0i32 }, core::ptr::null_mut::<*mut page>())
}

fn cpa_set_pages_array(pages: *mut *mut page, numpages: i32, mask: pgprot) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set_clr(core::ptr::null_mut::<u64>(), numpages, mask, unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32, 4i32, pages)
}

fn cpa_clear_pages_array(pages: *mut *mut page, numpages: i32, mask: pgprot) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set_clr(core::ptr::null_mut::<u64>(), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, mask, 0i32, 4i32, pages)
}

#[unsafe(no_mangle)]
extern "C" fn _set_memory_uc(mut addr: u64, numpages: i32) -> i32 {
// for now UC MINUS. see comments in ioremap()
// If you really need strong UC use ioremap_uc(), but note
// that you cannot override IO areas with set_memory_*() as
// these helpers cannot work with IO memory.
change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, cachemode2pgprot(2u32), 0i32)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_memory_uc")]
extern "C" fn set_memory_uc(addr: u64, numpages: i32) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// for now UC MINUS. see comments in ioremap()
*__slate_slot_0 = memtype_reserve(__pa!(addr), __pa!(addr).wrapping_add((numpages as u64).wrapping_mul(_AC!() << 12i32)), 2u32, core::ptr::null_mut::<u32>());
if *__slate_slot_0 == 0 {
*__slate_slot_0 = _set_memory_uc(addr, numpages);
if *__slate_slot_0 != 0 {
memtype_free(__pa!(addr), __pa!(addr).wrapping_add((numpages as u64).wrapping_mul(_AC!() << 12i32)));
} else {
return 0;
}
}
return *__slate_slot_0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_memory_uc_633: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, i32) -> i32>, *mut ()>(Some(set_memory_uc)) };

#[unsafe(no_mangle)]
extern "C" fn _set_memory_wc(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut ret: i32 = change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, cachemode2pgprot(2u32), 0i32);
if ret == 0 {
ret = change_page_attr_set_clr(core::ptr::addr_of_mut!(addr), numpages, cachemode2pgprot(1u32), unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 7i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32, 0i32, core::ptr::null_mut::<*mut page>());
}
ret
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_memory_wc")]
extern "C" fn set_memory_wc(addr: u64, numpages: i32) -> i32 {
let mut ret: i32 = unsafe { memtype_reserve(__pa!(addr), __pa!(addr).wrapping_add((numpages as u64).wrapping_mul(_AC!() << 12i32)), 1u32, core::ptr::null_mut::<u32>()) };
if ret != 0 {
return ret;
}
ret = _set_memory_wc(addr, numpages);
if ret != 0 {
unsafe { memtype_free(__pa!(addr), __pa!(addr).wrapping_add((numpages as u64).wrapping_mul(_AC!() << 12i32))) };
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_memory_wc_634: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, i32) -> i32>, *mut ()>(Some(set_memory_wc)) };

#[unsafe(no_mangle)]
extern "C" fn _set_memory_wt(mut addr: u64, numpages: i32) -> i32 {
change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, cachemode2pgprot(4u32), 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn _set_memory_wb(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
// WB cache mode is hard wired to all cache attribute bits being 0
change_page_attr_clear(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 7i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_memory_wb")]
extern "C" fn set_memory_wb(addr: u64, numpages: i32) -> i32 {
let ret: i32 = _set_memory_wb(addr, numpages);
if ret != 0 {
return ret;
}
unsafe { memtype_free(__pa!(addr), __pa!(addr).wrapping_add((numpages as u64).wrapping_mul(_AC!() << 12i32))) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_memory_wb_635: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, i32) -> i32>, *mut ()>(Some(set_memory_wb)) };

/// Prevent speculative access to a page by marking it not-present
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_mce_nospec")]
extern "C" fn set_mce_nospec(pfn: u64) -> i32 {
let mut decoy_addr: u64 = 0;
let mut rc: i32 = 0;
// SGX pages are not in the 1:1 map
if arch_is_platform_page(pfn << 12i32) {
return 0;
}
// We would like to just call:
//      set_memory_XX((unsigned long)pfn_to_kaddr(pfn), 1);
// but doing that would radically increase the odds of a
// speculative access to the poison page because we'd have
// the virtual address of the kernel 1:1 mapping sitting
// around in registers.
// Instead we get tricky.  We create a non-canonical address
// that looks just like the one we want, but has bit 63 flipped.
// This relies on set_memory_XX() properly sanitizing any __pa()
// results with __PHYSICAL_MASK or PTE_PFN_MASK.
decoy_addr = (pfn << 12i32).wrapping_add((unsafe { page_offset_base }) ^ BIT!(63i32));
rc = set_memory_np(decoy_addr, 1i32);
if rc != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014Could not invalidate pfn=0x%lx from 1:1 map\n".as_ptr() as *const u8, pfn) };
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_mce_nospec_636: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64) -> i32>, *mut ()>(Some(set_mce_nospec)) };

/// Restore full speculative operation to the pfn.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.clear_mce_nospec")]
extern "C" fn clear_mce_nospec(pfn: u64) -> i32 {
let addr: u64 = pfn_to_kaddr(pfn) as u64;
set_memory_p(addr, 1i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_clear_mce_nospec_637: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64) -> i32>, *mut ()>(Some(clear_mce_nospec)) };

#[unsafe(no_mangle)]
extern "C" fn set_memory_x(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
if (unsafe { __supported_pte_mask }) & _AT!(u64, 1i32) << 63i32 == 0 {
return 0;
}
change_page_attr_clear(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 63i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_nx(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
if (unsafe { __supported_pte_mask }) & _AT!(u64, 1i32) << 63i32 == 0 {
return 0;
}
change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 63i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_ro(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_clear(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 6i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_rox(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: pgprot = unsafe { core::mem::zeroed() };
let mut clr: pgprot = unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 6i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
if (unsafe { __supported_pte_mask }) & _AT!(u64, 1i32) << 63i32 != 0 {
let __v1: u64 = clr.pgprot;
let __v2: u64 = __v1 | _AT!(u64, 1i32) << 63i32;
clr.pgprot = __v2;
}
change_page_attr_set_clr(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_3 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_3)
} }, clr, 0i32, 16i32, core::ptr::null_mut::<*mut page>())
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_rw(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 1i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_np(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_clear(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_np_noalias(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set_clr(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, unsafe { *{
__slate_compound_1 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 0i32, 8i32, core::ptr::null_mut::<*mut page>())
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_p(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_4k(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set_clr(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 11i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, unsafe { *{
__slate_compound_1 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 1i32, 0i32, core::ptr::null_mut::<*mut page>())
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_nonglobal(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_clear(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 8i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_memory_global(mut addr: u64, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
change_page_attr_set(core::ptr::addr_of_mut!(addr), numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 8i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }, 0i32)
}

/// __set_memory_enc_pgtable() is used for the hypervisors that get
/// informed about "encryption" status via page tables.
fn __set_memory_enc_pgtable(mut addr: u64, numpages: i32, enc: bool) -> i32 {
let mut __slate_compound_21: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: pgprot = unsafe { core::mem::zeroed() };
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v2: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v2) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __v6: core::mem::MaybeUninit<pgprot> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut pgprot = core::ptr::addr_of_mut!(__v6) as *mut pgprot;
let mut __v7: core::mem::MaybeUninit<pgprot> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut pgprot = core::ptr::addr_of_mut!(__v7) as *mut pgprot;
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v12: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v12) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut cpa: core::mem::MaybeUninit<cpa_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut cpa_data = core::ptr::addr_of_mut!(cpa) as *mut cpa_data;
let mut empty: core::mem::MaybeUninit<pgprot> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut pgprot = core::ptr::addr_of_mut!(empty) as *mut pgprot;
unsafe {
core::ptr::write(__slate_slot_18, *{
__slate_compound_27 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_27)
});
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 72usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
__ret_warn_on_1 = (addr & !(!(_AC!() << 12i32).wrapping_sub(1)) != 0) as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"misaligned address: %#lx\n".as_ptr() as *const u8, addr);
*__slate_slot_12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string___set_memory_enc_pgtable_25, sym __slate_asm_string___set_memory_enc_pgtable_26, const 2430 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_10 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
(*__slate_slot_12).unwrap()(*__slate_slot_10, addr);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
// Should not be working on unaligned addresses
let __v15: i32 = __ret_warn_on_1;
if __v15 != 0 {
let __v9: u64 = addr;
addr = __v9 & !(_AC!() << 12i32).wrapping_sub(1);
}
memset(core::ptr::addr_of_mut!(*__slate_slot_17) as *mut (), 0i32, core::mem::size_of::<cpa_data>() as u64);
(*__slate_slot_17).vaddr = core::ptr::addr_of_mut!(addr);
(*__slate_slot_17).numpages = numpages as u64;
if enc {
*__slate_slot_7 = *{
__slate_compound_23 = pgprot { pgprot: cc_mkenc(pgprot_val!(*__slate_slot_18)) };
core::ptr::addr_of_mut!(__slate_compound_23)
};
} else {
*__slate_slot_7 = *{
__slate_compound_24 = pgprot { pgprot: cc_mkdec(pgprot_val!(*__slate_slot_18)) };
core::ptr::addr_of_mut!(__slate_compound_24)
};
}
(*__slate_slot_17).mask_set = *__slate_slot_7;
if enc {
*__slate_slot_6 = *{
__slate_compound_21 = pgprot { pgprot: cc_mkdec(pgprot_val!(*__slate_slot_18)) };
core::ptr::addr_of_mut!(__slate_compound_21)
};
} else {
*__slate_slot_6 = *{
__slate_compound_22 = pgprot { pgprot: cc_mkenc(pgprot_val!(*__slate_slot_18)) };
core::ptr::addr_of_mut!(__slate_compound_22)
};
}
(*__slate_slot_17).mask_clr = *__slate_slot_6;
(*__slate_slot_17).pgd = init_mm.__slate_anon_0.pgd;
// Must avoid aliasing mappings in the highmem code
kmap_flush_unused();
vm_unmap_aliases();
// Flush the caches as needed before changing the encryption attribute.
if unsafe { x86_platform.guest.enc_tlb_flush_required }.unwrap()(enc) {
cpa_flush(core::ptr::addr_of_mut!(*__slate_slot_17), unsafe { x86_platform.guest.enc_cache_flush_required }.unwrap()() as i32);
}
// Notify hypervisor that we are about to set/clr encryption attribute.
*__slate_slot_16 = unsafe { x86_platform.guest.enc_status_change_prepare }.unwrap()(addr, numpages, enc);
if *__slate_slot_16 == 0 {
*__slate_slot_16 = __change_page_attr_set_clr(core::ptr::addr_of_mut!(*__slate_slot_17), 1i32);
// After changing the encryption attribute, we need to flush TLBs again
// in case any speculative TLB caching occurred (but no need to flush
// caches again).  We could just use cpa_flush_all(), but in case TLB
// flushing gets optimized in the cpa_flush() path use the same logic
// as above.
cpa_flush(core::ptr::addr_of_mut!(*__slate_slot_17), 0i32);
if *__slate_slot_16 != 0 {
return *__slate_slot_16;
} else {
// Notify hypervisor that we have successfully set/clr encryption attribute.
*__slate_slot_16 = unsafe { x86_platform.guest.enc_status_change_finish }.unwrap()(addr, numpages, enc);
if *__slate_slot_16 == 0 {
return 0;
}
}
}
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"CPA VMM failure to convert memory (addr=%p, numpages=%d) to %s: %d\n".as_ptr() as *const u8, addr as *mut (), numpages, if enc { c"private".as_ptr() as *mut u8 } else { c"shared".as_ptr() as *mut u8 }, *__slate_slot_16);
*__slate_slot_2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string___set_memory_enc_pgtable_19, sym __slate_asm_string___set_memory_enc_pgtable_20, const 2476 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_0 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_2).unwrap()(*__slate_slot_0, addr as *mut (), numpages, if enc { c"private".as_ptr() as *mut u8 } else { c"shared".as_ptr() as *mut u8 }, *__slate_slot_16);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
return *__slate_slot_16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_638: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_639: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// The lock serializes conversions between private and shared memory.
///
/// It is taken for read on conversion. A write lock guarantees that no
/// concurrent conversions are in progress.
static mut mem_enc_lock: rw_semaphore = rw_semaphore { count: __SlateRecord_types_h_197 { counter: 0i64 }, owner: __SlateRecord_types_h_197 { counter: 0i64 }, osq: optimistic_spin_queue { tail: __SlateRecord_types_h_190 { counter: 0i32 } }, wait_lock: raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } }, first_waiter: core::ptr::null_mut::<rwsem_waiter>() };

/// Stop new private<->shared conversions.
///
/// Taking the exclusive mem_enc_lock waits for in-flight conversions to complete.
/// The lock is not released to prevent new conversions from being started.
#[unsafe(no_mangle)]
extern "C" fn set_memory_enc_stop_conversion() -> bool {
// In a crash scenario, sleep is not allowed. Try to take the lock.
// Failure indicates that there is a race with the conversion.
if (unsafe { oops_in_progress }) != 0 {
return (unsafe { down_write_trylock(core::ptr::addr_of_mut!(mem_enc_lock)) }) != 0;
}
unsafe { down_write(core::ptr::addr_of_mut!(mem_enc_lock)) };
true
}

fn __set_memory_enc_dec(addr: u64, numpages: i32, enc: bool) -> i32 {
let mut ret: i32 = 0;
if cc_platform_has(0u32) {
if (unsafe { down_read_trylock(core::ptr::addr_of_mut!(mem_enc_lock)) }) == 0 {
return 16i32.wrapping_neg();
}
ret = __set_memory_enc_pgtable(addr, numpages, enc);
unsafe { up_read(core::ptr::addr_of_mut!(mem_enc_lock)) };
}
ret
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_memory_encrypted")]
extern "C" fn set_memory_encrypted(addr: u64, numpages: i32) -> i32 {
__set_memory_enc_dec(addr, numpages, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_memory_encrypted_640: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, i32) -> i32>, *mut ()>(Some(set_memory_encrypted)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_memory_decrypted")]
extern "C" fn set_memory_decrypted(addr: u64, numpages: i32) -> i32 {
__set_memory_enc_dec(addr, numpages, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_memory_decrypted_641: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, i32) -> i32>, *mut ()>(Some(set_memory_decrypted)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_pages_uc")]
extern "C" fn set_pages_uc(page: *mut page, numpages: i32) -> i32 {
let addr: u64 = page_address!(page) as u64;
set_memory_uc(addr, numpages)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_pages_uc_642: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, i32) -> i32>, *mut ()>(Some(set_pages_uc)) };

fn _set_pages_array(pages: *mut *mut page, numpages: i32, new_type: u32) -> i32 {
let mut __slate_compound_10: pgprot = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut free_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(free_idx) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut set_type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(set_type) as *mut u32;
let mut end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(end) as *mut u64;
let mut start: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(start) as *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_6 = 0;
'__join_5: {
while *__slate_slot_6 < numpages {
if PageHighMem(*pages.wrapping_offset(*__slate_slot_6 as isize) as *const page) == 0 {
*__slate_slot_9 = ((unsafe { *pages.wrapping_offset(*__slate_slot_6 as isize) }.offset_from(vmemmap_base as *mut page) as i64) as u64) << 12i32;
*__slate_slot_8 = (*__slate_slot_9).wrapping_add(_AC!() << 12i32);
if memtype_reserve(*__slate_slot_9, *__slate_slot_8, new_type, core::ptr::null_mut::<u32>()) != 0 {
break '__join_5;
}
}
let __v3: i32 = *__slate_slot_6;
*__slate_slot_6 = __v3.wrapping_add(1);
}
// If WC, set to UC- first and then WC
*__slate_slot_7 = (if (new_type as u32) == 1 { 2u32 } else { new_type as u32 }) as u32;
*__slate_slot_4 = cpa_set_pages_array(pages, numpages, cachemode2pgprot(*__slate_slot_7));
if *__slate_slot_4 == 0 && (new_type as u32) == 1 {
*__slate_slot_4 = change_page_attr_set_clr(core::ptr::null_mut::<u64>(), numpages, cachemode2pgprot(1u32), *{
__slate_compound_10 = pgprot { pgprot: _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 7i32 };
core::ptr::addr_of_mut!(__slate_compound_10)
}, 0i32, 4i32, pages);
}
if *__slate_slot_4 == 0 {
return 0;
}
}
*__slate_slot_5 = *__slate_slot_6;
*__slate_slot_6 = 0;
while *__slate_slot_6 < *__slate_slot_5 {
if PageHighMem(*pages.wrapping_offset(*__slate_slot_6 as isize) as *const page) == 0 {
*__slate_slot_9 = ((unsafe { *pages.wrapping_offset(*__slate_slot_6 as isize) }.offset_from(vmemmap_base as *mut page) as i64) as u64) << 12i32;
*__slate_slot_8 = (*__slate_slot_9).wrapping_add(_AC!() << 12i32);
memtype_free(*__slate_slot_9, *__slate_slot_8);
}
let __v1: i32 = *__slate_slot_6;
*__slate_slot_6 = __v1.wrapping_add(1);
}
return 22i32.wrapping_neg();
}
// Success
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_pages_array_uc")]
extern "C" fn set_pages_array_uc(pages: *mut *mut page, numpages: i32) -> i32 {
_set_pages_array(pages, numpages, 2u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_pages_array_uc_643: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut *mut page, i32) -> i32>, *mut ()>(Some(set_pages_array_uc)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_pages_array_wc")]
extern "C" fn set_pages_array_wc(pages: *mut *mut page, numpages: i32) -> i32 {
_set_pages_array(pages, numpages, 1u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_pages_array_wc_644: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut *mut page, i32) -> i32>, *mut ()>(Some(set_pages_array_wc)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_pages_wb")]
extern "C" fn set_pages_wb(page: *mut page, numpages: i32) -> i32 {
let addr: u64 = page_address!(page) as u64;
set_memory_wb(addr, numpages)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_pages_wb_645: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, i32) -> i32>, *mut ()>(Some(set_pages_wb)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.set_memory.set_pages_array_wb")]
extern "C" fn set_pages_array_wb(pages: *mut *mut page, numpages: i32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut retval: i32 = 0;
let mut start: u64 = 0;
let mut end: u64 = 0;
// WB cache mode is hard wired to all cache attribute bits being 0
retval = cpa_clear_pages_array(pages, numpages, unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 7i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} });
if retval != 0 {
return retval;
}
for i in 0..numpages {
if PageHighMem((unsafe { *pages.wrapping_offset(i as isize) }) as *const page) == 0 {
start = (((unsafe { unsafe { *pages.wrapping_offset(i as isize) }.offset_from(vmemmap_base as *mut page) }) as i64) as u64) << 12i32;
end = start.wrapping_add(_AC!() << 12i32);
unsafe { memtype_free(start, end) };
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_set_pages_array_wb_646: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut *mut page, i32) -> i32>, *mut ()>(Some(set_pages_array_wb)) };

#[unsafe(no_mangle)]
extern "C" fn set_pages_ro(page: *mut page, numpages: i32) -> i32 {
let addr: u64 = page_address!(page) as u64;
set_memory_ro(addr, numpages)
}

#[unsafe(no_mangle)]
extern "C" fn set_pages_rw(page: *mut page, numpages: i32) -> i32 {
let addr: u64 = page_address!(page) as u64;
set_memory_rw(addr, numpages)
}

fn __set_pages_p(page: *mut page, numpages: i32, cpa_flags: u32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
let mut tempaddr: u64 = page_address!(page) as u64;
let mut cpa: cpa_data = {
let mut __t0: cpa_data = unsafe { core::mem::zeroed() };
__t0.vaddr = core::ptr::addr_of_mut!(tempaddr);
__t0.pgd = core::ptr::null_mut::<__SlateRecord_pgtable_types_h_295>();
__t0.mask_set = unsafe { *{
__slate_compound_0 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
__t0.mask_clr = unsafe { *{
__slate_compound_1 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
__t0.numpages = numpages as u64;
__t0.flags = 8 | cpa_flags;
__t0
};
// No alias checking needed for setting present flag. otherwise,
// we may need to break large pages for 64-bit kernel text
// mappings (this adds to complexity if we want to do this from
// atomic context especially). Let's keep it simple!
__change_page_attr_set_clr(core::ptr::addr_of_mut!(cpa), 1i32)
}

fn __set_pages_np(page: *mut page, numpages: i32, cpa_flags: u32) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
let mut tempaddr: u64 = page_address!(page) as u64;
let mut cpa: cpa_data = {
let mut __t0: cpa_data = unsafe { core::mem::zeroed() };
__t0.vaddr = core::ptr::addr_of_mut!(tempaddr);
__t0.pgd = core::ptr::null_mut::<__SlateRecord_pgtable_types_h_295>();
__t0.mask_set = unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
__t0.mask_clr = unsafe { *{
__slate_compound_1 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 6i32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
__t0.numpages = numpages as u64;
__t0.flags = 8 | cpa_flags;
__t0
};
// No alias checking needed for setting not present flag. otherwise,
// we may need to break large pages for 64-bit kernel text
// mappings (this adds to complexity if we want to do this from
// atomic context especially). Let's keep it simple!
__change_page_attr_set_clr(core::ptr::addr_of_mut!(cpa), 1i32)
}

#[unsafe(no_mangle)]
extern "C" fn set_direct_map_invalid_noflush(page: *mut page) -> i32 {
__set_pages_np(page, 1i32, 0u32)
}

#[unsafe(no_mangle)]
extern "C" fn set_direct_map_default_noflush(page: *mut page) -> i32 {
__set_pages_p(page, 1i32, 0u32)
}

#[unsafe(no_mangle)]
extern "C" fn set_direct_map_valid_noflush(page: *mut page, nr: u32, valid: bool) -> i32 {
if valid {
return __set_pages_p(page, nr as i32, 0u32);
}
__set_pages_np(page, nr as i32, 0u32)
}

#[unsafe(no_mangle)]
extern "C" fn kernel_page_present(page: *mut page) -> bool {
let mut level: u32 = 0;
if PageHighMem(page as *const page) != 0 {
return false;
}
let pte: *mut __SlateRecord_pgtable_64_types_h_21 = lookup_address(page_address!(page) as u64, core::ptr::addr_of_mut!(level));
pte_val!(unsafe { *pte }) & _AT!(u64, 1i32) << 0i32 != 0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn kernel_map_pages_in_pgd(pgd: *mut __SlateRecord_pgtable_types_h_295, pfn: u64, mut address: u64, numpages: u32, page_flags: u64) -> i32 {
let mut __slate_compound_8: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: pgprot = unsafe { core::mem::zeroed() };
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v2: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v2) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut cpa: core::mem::MaybeUninit<cpa_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut cpa_data = core::ptr::addr_of_mut!(cpa) as *mut cpa_data;
let mut retval: i32 = unsafe { core::mem::zeroed() };
unsafe {
retval = 22i32.wrapping_neg();
core::ptr::write(__slate_slot_6, {
let mut __t1: cpa_data = core::mem::zeroed();
__t1.vaddr = core::ptr::addr_of_mut!(address);
__t1.pgd = pgd;
__t1.mask_set = *{
__slate_compound_12 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_12)
};
__t1.mask_clr = *{
__slate_compound_13 = pgprot { pgprot: !page_flags & (_AT!(u64, 1i32) << 63i32 | _AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 6i32) };
core::ptr::addr_of_mut!(__slate_compound_13)
};
__t1.numpages = numpages as u64;
__t1.pfn = pfn;
__t1.flags = 8u32;
__t1
});
let __ret_warn_on: i32 = (num_online_cpus() > 1) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"Don't call after initializing SMP".as_ptr() as *const u8);
*__slate_slot_2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_kernel_map_pages_in_pgd_10, sym __slate_asm_string_kernel_map_pages_in_pgd_11, const 2775 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_0 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_2).unwrap()(*__slate_slot_0);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
if __supported_pte_mask & _AT!(u64, 1i32) << 63i32 != 0 {
if page_flags & _AT!(u64, 0u64) == 0 {
(*__slate_slot_6).mask_clr = *{
__slate_compound_9 = pgprot { pgprot: cc_mkenc(pgprot_val!((*__slate_slot_6).mask_clr)) };
core::ptr::addr_of_mut!(__slate_compound_9)
};
}
(*__slate_slot_6).mask_set = *{
__slate_compound_8 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 | page_flags };
core::ptr::addr_of_mut!(__slate_compound_8)
};
retval = __change_page_attr_set_clr(core::ptr::addr_of_mut!(*__slate_slot_6), 1i32);
__flush_tlb_all();
}
return retval;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_647: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// __flush_tlb_all() flushes mappings only on current CPU and hence this
/// function shouldn't be used in an SMP environment. Presently, it's used only
/// during boot (way before smp_init()) by EFI subsystem and hence is ok.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn kernel_unmap_pages_in_pgd(pgd: *mut __SlateRecord_pgtable_types_h_295, mut address: u64, numpages: u64) -> i32 {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
let mut retval: i32 = 0;
// The typical sequence for unmapping is to find a pte through
// lookup_address_in_pgd() (ideally, it should never return NULL because
// the address is already mapped) and change its protections. As pfn is
// the *target* of a mapping, it's not useful while unmapping.
let mut cpa: cpa_data = {
let mut __t0: cpa_data = unsafe { core::mem::zeroed() };
__t0.vaddr = core::ptr::addr_of_mut!(address);
__t0.pgd = pgd;
__t0.mask_set = unsafe { *{
__slate_compound_0 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
__t0.mask_clr = unsafe { *{
__slate_compound_1 = pgprot { pgprot: _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | _AT!(u64, 1i32) << 6i32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
__t0.numpages = numpages;
__t0.pfn = 0u64;
__t0.flags = 8u32;
__t0
};
let mut __v2: i32;
let __ret_warn_on: i32 = (num_online_cpus() > 1) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Don't call after initializing SMP".as_ptr() as *const u8) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_kernel_unmap_pages_in_pgd_5, sym __slate_asm_string_kernel_unmap_pages_in_pgd_6, const 2818 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = __ret_warn_on;
retval = __change_page_attr_set_clr(core::ptr::addr_of_mut!(cpa), 1i32);
unsafe { __flush_tlb_all() };
retval
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_648: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// The testcases use internal knowledge of the implementation that shouldn't
// be exposed to the rest of the kernel. Include these directly here.
static __slate_asm_string___change_page_attr_set_clr_11: [u8; 1] = *b"\0";

static __slate_asm_string___change_page_attr_set_clr_12: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string___cpa_process_fault_3: [u8; 58] = *b"\x014CPA: called for zero pte. vaddr = %lx cpa->vaddr = %lx\n\0";

static __slate_asm_string___cpa_process_fault_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string___set_memory_enc_pgtable_19: [u8; 68] = *b"CPA VMM failure to convert memory (addr=%p, numpages=%d) to %s: %d\n\0";

static __slate_asm_string___set_memory_enc_pgtable_20: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string___set_memory_enc_pgtable_25: [u8; 26] = *b"misaligned address: %#lx\n\0";

static __slate_asm_string___set_memory_enc_pgtable_26: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string___should_split_large_page_10: [u8; 1] = *b"\0";

static __slate_asm_string___should_split_large_page_11: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_change_page_attr_set_clr_19: [u8; 1] = *b"\0";

static __slate_asm_string_change_page_attr_set_clr_20: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_change_page_attr_set_clr_21: [u8; 1] = *b"\0";

static __slate_asm_string_change_page_attr_set_clr_22: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_cpa_flush_10: [u8; 1] = *b"\0";

static __slate_asm_string_cpa_flush_11: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_cpa_flush_all_10: [u8; 1] = *b"\0";

static __slate_asm_string_cpa_flush_all_11: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_cpu_cache_invalidate_memregion_1: [u8; 1] = *b"\0";

static __slate_asm_string_cpu_cache_invalidate_memregion_2: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_kernel_map_pages_in_pgd_10: [u8; 34] = *b"Don't call after initializing SMP\0";

static __slate_asm_string_kernel_map_pages_in_pgd_11: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_kernel_unmap_pages_in_pgd_5: [u8; 34] = *b"Don't call after initializing SMP\0";

static __slate_asm_string_kernel_unmap_pages_in_pgd_6: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_pfn_pte_1: [u8; 1] = *b"\0";

static __slate_asm_string_pfn_pte_2: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/include/asm/pgtable.h\0";

static __slate_asm_string_pmd_free_0: [u8; 1] = *b"\0";

static __slate_asm_string_pmd_free_1: [u8; 59] = *b"/home/takashi/c-corpus/linux/include/asm-generic/pgalloc.h\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_1: [u8; 1] = *b"\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/include/linux/rwsem.h\0";

static __slate_asm_string_slow_virt_to_phys_0: [u8; 1] = *b"\0";

static __slate_asm_string_slow_virt_to_phys_1: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";

static __slate_asm_string_verify_rwx_3: [u8; 83] = *b"CPA detected W^X violation: %016llx -> %016llx range: 0x%016lx - 0x%016lx PFN %lx\n\0";

static __slate_asm_string_verify_rwx_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/arch/x86/mm/pat/set_memory.c\0";
