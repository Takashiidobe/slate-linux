
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! These attributes define new context lock (Clang: capability) types.
//! Internal only.
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! Turn individual warnings and errors on and off locally, depending
//! on version.
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! Copyright (C) 2006
//! NTT (Nippon Telegraph and Telephone Corporation).
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
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
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! @a is a power of 2 value
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! Internal kernel timer frequency
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! This counts to 15. Any more, it will return 16th argument.
//! To support Context Analysis, we need to allow the compiler to see the
//! acquisition and release of the context lock. However, the "cleanup" helpers
//! wrap the lock in a struct passed through separate helper functions, which
//! hides the lock alias from the compiler (no inter-procedural analysis).
//!
//! To make it work, we introduce an explicit alias to the context lock instance
//! that is "cleaned" up with a separate cleanup helper. This helper is a dummy
//! function that does nothing at runtime, but has the "_unlock" attribute to
//! tell the compiler what happens at the end of the scope.
//!
//! To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
//! to redefine the constructor, which then also creates the alias variable with
//! the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
//! used.
//!
//! Example usage:
//!
//!   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//!   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//!
//! Note: To support the for-loop based scoped helpers, the auxiliary variable
//! must be a pointer to the "class" type because it is defined in the same
//! statement as the guard variable. However, we initialize it with the lock
//! pointer (despite the type mismatch, the compiler's alias analysis still works
//! as expected). The "_unlock" attribute receives a pointer to the auxiliary
//! variable (a double pointer to the class type), and must be cast and
//! dereferenced appropriately.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
//! 4th level page in 5-level paging case
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
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
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
//! restart by calling sys_restart_syscall
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
//! Physical address zone modifiers (see linux/mmzone.h - low four bits)
//!
//! Do not put any conditional on these. If necessary modify the definitions
//! without the underscores and use them consistently. The definitions here may
//! be used in bit comparisons.
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
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! If a tracepoint needs to be called from a header file, it is not
//! recommended to call it directly, as tracepoints in header files
//! may cause side-effects and bloat the kernel. Instead, use
//! tracepoint_enabled() to test if the tracepoint is enabled, then if
//! it is, call a wrapper function defined in a C file that will then
//! call the tracepoint.
//!
//! For "trace_foo_bar()", you would need to create a wrapper function
//! in a C file to call trace_foo_bar():
//!   void do_trace_foo_bar(args) { trace_foo_bar(args); }
//! Then in the header file, declare the tracepoint:
//!   DECLARE_TRACEPOINT(foo_bar);
//! And call your wrapper:
//!   static inline void some_inlined_function() {
//!            [..]
//!            if (tracepoint_enabled(foo_bar))
//!                    do_trace_foo_bar(args);
//!            [..]
//!   }
//!
//! Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//!   but is safe to have in headers, where trace_foo_bar_enabled() is not.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
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
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
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
//! kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
//! kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
//! and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
//! Algorithm masks and types.
//! Autoloaded crypto modules should only use a prefixed name to avoid allowing
//! arbitrary modules to be loaded. Loading from userspace may still need the
//! unprefixed names, so retains those aliases as well.
//! This uses __MODULE_INFO directly instead of MODULE_ALIAS because pre-4.3
//! gcc (e.g. avr32 toolchain) uses __LINE__ for uniqueness, and this macro
//! expands twice on the same line. Instead, use a separate base name for the
//! alias.
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
// The "assert_capability" attribute is a bit confusingly named. It does not
// generate a check. Instead, it tells the analysis to *assume* the capability
// is held. This is used for augmenting runtime assertions, that can then help
// with patterns beyond the compiler's static reasoning abilities.
// __guarded_by() - struct member and globals attribute, declares variable
//                  only accessible within active context
// @...: context lock instance pointer(s)
//
// Declares that the struct member or global variable is only accessible within
// the context entered by the given context lock(s). Read operations on the data
// require shared access to at least one of the context locks, while write
// operations require exclusive access to all listed context locks.
//
// .. code-block:: c
//
// 	struct some_state {
// 		spinlock_t lock;
// 		long counter __guarded_by(&lock);
// 	};
//
// 	struct some_state {
// 		spinlock_t lock1, lock2;
// 		long counter __guarded_by(&lock1, &lock2);
// 	};
// context_lock_struct() - declare or define a context lock struct
// @name: struct name
//
// Helper to declare or define a struct type that is also a context lock.
//
// .. code-block:: c
//
// 	context_lock_struct(my_handle) {
// 		int foo;
// 		long bar;
// 	};
//
// 	struct some_state {
// 		...
// 	};
// 	// ... declared elsewhere ...
// 	context_lock_struct(some_state);
//
// Note: The implementation defines several helper functions that can acquire
// and release the context lock.
// disable_context_analysis() - disables context analysis
//
// Disables context analysis. Must be paired with a later
// enable_context_analysis().
// enable_context_analysis() - re-enables context analysis
//
// Re-enables context analysis. Must be paired with a prior
// disable_context_analysis().
// __no_context_analysis - function attribute, disables context analysis
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Prefer use of `context_unsafe()` where possible.
// context_unsafe() - disable context checking for contained code
//
// Disables context checking for contained statements or expression.
//
// .. code-block:: c
//
// 	struct some_data {
// 		spinlock_t lock;
// 		int counter __guarded_by(&lock);
// 	};
//
// 	int foo(struct some_data *d)
// 	{
// 		// ...
// 		// other code that is still checked ...
// 		// ...
// 		return context_unsafe(d->counter);
// 	}
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
// Note the missing underscores.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noinline-function-attribute
// clang: mentioned
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
// Severity used in pragma directives
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Atomic operations are already serializing on x86
// list_first_entry - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note, that list is expected to be not empty.
// list_next_entry - get the next element in list
// @pos:	the type * to cursor
// @member:	the name of the list_head within the struct.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_entry_is_head - test if the entry points to the head of the list
// @pos:	the type * to cursor
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// No such file or directory
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Text file busy
// Operation not supported on transport endpoint
// Operation now in progress
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
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
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
// Operation is not supported
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
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// re-arm the timer
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// disable sas during sighandling
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
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
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
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
macro_rules! CAMELLIA_F {
($xl:expr, $xr:expr, $kl:expr, $kr:expr, $yl:expr, $yr:expr, $il:expr, $ir:expr, $t0:expr, $t1:expr) => {
let mut __v: u32;
$il = ((($xl) as i64) ^ $kl) as u32;
$ir = ((($xr) as i64) ^ $kr) as u32;
$t0 = ($il) >> 16i32;
$t1 = ($ir) >> 16i32;
$yl = (unsafe { camellia_sp1110.0[((($ir) as u8) as i32) as usize] }) ^ unsafe { camellia_sp0222.0[(((($t1) >> 8i32) as u8) as i32) as usize] } ^ unsafe { camellia_sp3033.0[((($t1) as u8) as i32) as usize] } ^ unsafe { camellia_sp4404.0[(((($ir) >> 8i32) as u8) as i32) as usize] };
$yr = (unsafe { camellia_sp1110.0[(((($t0) >> 8i32) as u8) as i32) as usize] }) ^ unsafe { camellia_sp0222.0[((($t0) as u8) as i32) as usize] } ^ unsafe { camellia_sp3033.0[(((($il) >> 8i32) as u8) as i32) as usize] } ^ unsafe { camellia_sp4404.0[((($il) as u8) as i32) as usize] };
let __v2: u32 = $yl;
let __v3: u32 = __v2 ^ $yr;
$yl = __v3;
$yr = ror32($yr, 8u32);
let __v4: u32 = $yr;
let __v5: u32 = __v4 ^ $yl;
$yr = __v5;
__v = __v5;
};
}

macro_rules! FLS {
($i:expr, $t0:expr, $subkey:expr, $il:expr, $io:expr, $ir:expr, $t1:expr) => {
let mut __v: u32;
let mut __v2: u32;
$t0 = SUBKEY_L!($i.wrapping_add(1), $subkey);
$il = SUBKEY_R!($i.wrapping_add(0), $subkey);
let __v3: u32 = $t0;
let __v4: u32 = __v3 & unsafe { *$io.wrapping_offset(0 as isize) };
$t0 = __v4;
let __v5: u32 = $il;
let __v6: u32 = __v5 | unsafe { *$io.wrapping_offset(3 as isize) };
$il = __v6;
let __v7: *mut u32 = $io.wrapping_offset(2 as isize);
let __v8: u32 = unsafe { *__v7 };
let __v9: u32 = __v8 ^ $il;
unsafe {
*__v7 = __v9;
}
let __v10: u32 = rol32($t0, 1u32);
let __v11: *mut u32 = $io.wrapping_offset(1 as isize);
unsafe {
*__v11 = *__v11 ^ __v10;
}
$ir = SUBKEY_L!($i.wrapping_add(0), $subkey);
$t1 = SUBKEY_R!($i.wrapping_add(1), $subkey);
let __v14: u32 = $ir;
let __v15: u32 = __v14 & unsafe { *$io.wrapping_offset(2 as isize) };
$ir = __v15;
let __v16: u32 = $t1;
let __v17: u32 = __v16 | unsafe { *$io.wrapping_offset(1 as isize) };
$t1 = __v17;
let __v18: *mut u32 = $io.wrapping_offset(0 as isize);
let __v19: u32 = unsafe { *__v18 };
let __v20: u32 = __v19 ^ $t1;
unsafe {
*__v18 = __v20;
}
let __v21: u32 = rol32($ir, 1u32);
let __v22: *mut u32 = $io.wrapping_offset(3 as isize);
let __v24: u32 = (unsafe { *__v22 }) ^ __v21;
unsafe {
*__v22 = __v24;
}
__v2 = __v24;
__v = __v2;
};
}

macro_rules! FLS__2 {
($i:expr, $t0:expr, $subkey:expr, $il:expr, $io:expr, $ir:expr, $t1:expr) => {
let mut __v: u32;
let mut __v2: u32;
$t0 = SUBKEY_L!($i.wrapping_add(0), $subkey);
$il = SUBKEY_R!($i.wrapping_add(1), $subkey);
let __v3: u32 = $t0;
let __v4: u32 = __v3 & unsafe { *$io.wrapping_offset(0 as isize) };
$t0 = __v4;
let __v5: u32 = $il;
let __v6: u32 = __v5 | unsafe { *$io.wrapping_offset(3 as isize) };
$il = __v6;
let __v7: *mut u32 = $io.wrapping_offset(2 as isize);
let __v8: u32 = unsafe { *__v7 };
let __v9: u32 = __v8 ^ $il;
unsafe {
*__v7 = __v9;
}
let __v10: u32 = rol32($t0, 1u32);
let __v11: *mut u32 = $io.wrapping_offset(1 as isize);
unsafe {
*__v11 = *__v11 ^ __v10;
}
$ir = SUBKEY_L!($i.wrapping_add(1), $subkey);
$t1 = SUBKEY_R!($i.wrapping_add(0), $subkey);
let __v14: u32 = $ir;
let __v15: u32 = __v14 & unsafe { *$io.wrapping_offset(2 as isize) };
$ir = __v15;
let __v16: u32 = $t1;
let __v17: u32 = __v16 | unsafe { *$io.wrapping_offset(1 as isize) };
$t1 = __v17;
let __v18: *mut u32 = $io.wrapping_offset(0 as isize);
let __v19: u32 = unsafe { *__v18 };
let __v20: u32 = __v19 ^ $t1;
unsafe {
*__v18 = __v20;
}
let __v21: u32 = rol32($ir, 1u32);
let __v22: *mut u32 = $io.wrapping_offset(3 as isize);
let __v24: u32 = (unsafe { *__v22 }) ^ __v21;
unsafe {
*__v22 = __v24;
}
__v2 = __v24;
__v = __v2;
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

/// macros
macro_rules! ROLDQ {
($ll:expr, $lr:expr, $rl:expr, $rr:expr, $w0:expr, $bits:expr) => {
let mut __v: u32;
$w0 = $ll;
$ll = (($ll) << $bits).wrapping_add(($lr) >> 32i32.wrapping_sub($bits));
$lr = (($lr) << $bits).wrapping_add(($rl) >> 32i32.wrapping_sub($bits));
$rl = (($rl) << $bits).wrapping_add(($rr) >> 32i32.wrapping_sub($bits));
let __v2: u32 = (($rr) << $bits).wrapping_add(($w0) >> 32i32.wrapping_sub($bits));
$rr = __v2;
__v = __v2;
};
}

macro_rules! ROLDQo32 {
($ll:expr, $lr:expr, $rl:expr, $rr:expr, $w0:expr, $w1:expr, $bits:expr) => {
let mut __v: u32;
$w0 = $ll;
$w1 = $lr;
$ll = (($lr) << $bits.wrapping_sub(32)).wrapping_add(($rl) >> 64i32.wrapping_sub($bits));
$lr = (($rl) << $bits.wrapping_sub(32)).wrapping_add(($rr) >> 64i32.wrapping_sub($bits));
$rl = (($rr) << $bits.wrapping_sub(32)).wrapping_add(($w0) >> 64i32.wrapping_sub($bits));
let __v2: u32 = (($w0) << $bits.wrapping_sub(32)).wrapping_add(($w1) >> 64i32.wrapping_sub($bits));
$rr = __v2;
__v = __v2;
};
}

macro_rules! ROUNDS {
($i:expr, $io:expr, $subkey:expr, $ir:expr, $il:expr) => {
let mut __v: u32;
let mut __v2: u32;
let __v3: *mut u32 = $io.wrapping_offset(2 as isize);
let __v4: u32 = unsafe { *__v3 };
let __v5: u32 = __v4 ^ SUBKEY_L!($i.wrapping_add(2), $subkey);
unsafe {
*__v3 = __v5;
}
let __v6: *mut u32 = $io.wrapping_offset(3 as isize);
let __v7: u32 = unsafe { *__v6 };
let __v8: u32 = __v7 ^ SUBKEY_R!($i.wrapping_add(2), $subkey);
unsafe {
*__v6 = __v8;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(1 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(0 as isize) >> 24i32) as u8) as i32) as usize] };
let __v9: u32 = $ir;
let __v10: u32 = __v9 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(1 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v10;
let __v11: u32 = $il;
let __v12: u32 = __v11 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(0 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v12;
let __v13: u32 = $ir;
let __v14: u32 = __v13 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(1 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v14;
let __v15: u32 = $il;
let __v16: u32 = __v15 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(0 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v16;
let __v17: u32 = $ir;
let __v18: u32 = __v17 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(1 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v18;
let __v19: u32 = $il;
let __v20: u32 = __v19 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(0 as isize) as u8) as i32) as usize] };
$il = __v20;
let __v21: u32 = $ir;
let __v22: u32 = __v21 ^ $il;
$ir = __v22;
let __v23: *mut u32 = $io.wrapping_offset(2 as isize);
let __v24: u32 = unsafe { *__v23 };
let __v25: u32 = __v24 ^ $ir;
unsafe {
*__v23 = __v25;
}
let __v26: u32 = ror32($il, 8u32) ^ $ir;
let __v27: *mut u32 = $io.wrapping_offset(3 as isize);
let __v29: u32 = (unsafe { *__v27 }) ^ __v26;
unsafe {
*__v27 = __v29;
}
__v2 = __v29;
let mut __v30: u32;
let __v31: *mut u32 = $io.wrapping_offset(0 as isize);
let __v32: u32 = unsafe { *__v31 };
let __v33: u32 = __v32 ^ SUBKEY_L!($i.wrapping_add(3), $subkey);
unsafe {
*__v31 = __v33;
}
let __v34: *mut u32 = $io.wrapping_offset(1 as isize);
let __v35: u32 = unsafe { *__v34 };
let __v36: u32 = __v35 ^ SUBKEY_R!($i.wrapping_add(3), $subkey);
unsafe {
*__v34 = __v36;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(3 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(2 as isize) >> 24i32) as u8) as i32) as usize] };
let __v37: u32 = $ir;
let __v38: u32 = __v37 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(3 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v38;
let __v39: u32 = $il;
let __v40: u32 = __v39 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(2 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v40;
let __v41: u32 = $ir;
let __v42: u32 = __v41 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(3 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v42;
let __v43: u32 = $il;
let __v44: u32 = __v43 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(2 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v44;
let __v45: u32 = $ir;
let __v46: u32 = __v45 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(3 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v46;
let __v47: u32 = $il;
let __v48: u32 = __v47 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(2 as isize) as u8) as i32) as usize] };
$il = __v48;
let __v49: u32 = $ir;
let __v50: u32 = __v49 ^ $il;
$ir = __v50;
let __v51: *mut u32 = $io.wrapping_offset(0 as isize);
let __v52: u32 = unsafe { *__v51 };
let __v53: u32 = __v52 ^ $ir;
unsafe {
*__v51 = __v53;
}
let __v54: u32 = ror32($il, 8u32) ^ $ir;
let __v55: *mut u32 = $io.wrapping_offset(1 as isize);
let __v57: u32 = (unsafe { *__v55 }) ^ __v54;
unsafe {
*__v55 = __v57;
}
__v30 = __v57;
let mut __v58: u32;
let __v59: *mut u32 = $io.wrapping_offset(2 as isize);
let __v60: u32 = unsafe { *__v59 };
let __v61: u32 = __v60 ^ SUBKEY_L!($i.wrapping_add(4), $subkey);
unsafe {
*__v59 = __v61;
}
let __v62: *mut u32 = $io.wrapping_offset(3 as isize);
let __v63: u32 = unsafe { *__v62 };
let __v64: u32 = __v63 ^ SUBKEY_R!($i.wrapping_add(4), $subkey);
unsafe {
*__v62 = __v64;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(1 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(0 as isize) >> 24i32) as u8) as i32) as usize] };
let __v65: u32 = $ir;
let __v66: u32 = __v65 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(1 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v66;
let __v67: u32 = $il;
let __v68: u32 = __v67 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(0 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v68;
let __v69: u32 = $ir;
let __v70: u32 = __v69 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(1 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v70;
let __v71: u32 = $il;
let __v72: u32 = __v71 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(0 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v72;
let __v73: u32 = $ir;
let __v74: u32 = __v73 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(1 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v74;
let __v75: u32 = $il;
let __v76: u32 = __v75 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(0 as isize) as u8) as i32) as usize] };
$il = __v76;
let __v77: u32 = $ir;
let __v78: u32 = __v77 ^ $il;
$ir = __v78;
let __v79: *mut u32 = $io.wrapping_offset(2 as isize);
let __v80: u32 = unsafe { *__v79 };
let __v81: u32 = __v80 ^ $ir;
unsafe {
*__v79 = __v81;
}
let __v82: u32 = ror32($il, 8u32) ^ $ir;
let __v83: *mut u32 = $io.wrapping_offset(3 as isize);
let __v85: u32 = (unsafe { *__v83 }) ^ __v82;
unsafe {
*__v83 = __v85;
}
__v58 = __v85;
let mut __v86: u32;
let __v87: *mut u32 = $io.wrapping_offset(0 as isize);
let __v88: u32 = unsafe { *__v87 };
let __v89: u32 = __v88 ^ SUBKEY_L!($i.wrapping_add(5), $subkey);
unsafe {
*__v87 = __v89;
}
let __v90: *mut u32 = $io.wrapping_offset(1 as isize);
let __v91: u32 = unsafe { *__v90 };
let __v92: u32 = __v91 ^ SUBKEY_R!($i.wrapping_add(5), $subkey);
unsafe {
*__v90 = __v92;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(3 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(2 as isize) >> 24i32) as u8) as i32) as usize] };
let __v93: u32 = $ir;
let __v94: u32 = __v93 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(3 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v94;
let __v95: u32 = $il;
let __v96: u32 = __v95 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(2 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v96;
let __v97: u32 = $ir;
let __v98: u32 = __v97 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(3 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v98;
let __v99: u32 = $il;
let __v100: u32 = __v99 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(2 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v100;
let __v101: u32 = $ir;
let __v102: u32 = __v101 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(3 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v102;
let __v103: u32 = $il;
let __v104: u32 = __v103 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(2 as isize) as u8) as i32) as usize] };
$il = __v104;
let __v105: u32 = $ir;
let __v106: u32 = __v105 ^ $il;
$ir = __v106;
let __v107: *mut u32 = $io.wrapping_offset(0 as isize);
let __v108: u32 = unsafe { *__v107 };
let __v109: u32 = __v108 ^ $ir;
unsafe {
*__v107 = __v109;
}
let __v110: u32 = ror32($il, 8u32) ^ $ir;
let __v111: *mut u32 = $io.wrapping_offset(1 as isize);
let __v113: u32 = (unsafe { *__v111 }) ^ __v110;
unsafe {
*__v111 = __v113;
}
__v86 = __v113;
let mut __v114: u32;
let __v115: *mut u32 = $io.wrapping_offset(2 as isize);
let __v116: u32 = unsafe { *__v115 };
let __v117: u32 = __v116 ^ SUBKEY_L!($i.wrapping_add(6), $subkey);
unsafe {
*__v115 = __v117;
}
let __v118: *mut u32 = $io.wrapping_offset(3 as isize);
let __v119: u32 = unsafe { *__v118 };
let __v120: u32 = __v119 ^ SUBKEY_R!($i.wrapping_add(6), $subkey);
unsafe {
*__v118 = __v120;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(1 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(0 as isize) >> 24i32) as u8) as i32) as usize] };
let __v121: u32 = $ir;
let __v122: u32 = __v121 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(1 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v122;
let __v123: u32 = $il;
let __v124: u32 = __v123 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(0 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v124;
let __v125: u32 = $ir;
let __v126: u32 = __v125 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(1 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v126;
let __v127: u32 = $il;
let __v128: u32 = __v127 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(0 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v128;
let __v129: u32 = $ir;
let __v130: u32 = __v129 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(1 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v130;
let __v131: u32 = $il;
let __v132: u32 = __v131 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(0 as isize) as u8) as i32) as usize] };
$il = __v132;
let __v133: u32 = $ir;
let __v134: u32 = __v133 ^ $il;
$ir = __v134;
let __v135: *mut u32 = $io.wrapping_offset(2 as isize);
let __v136: u32 = unsafe { *__v135 };
let __v137: u32 = __v136 ^ $ir;
unsafe {
*__v135 = __v137;
}
let __v138: u32 = ror32($il, 8u32) ^ $ir;
let __v139: *mut u32 = $io.wrapping_offset(3 as isize);
let __v141: u32 = (unsafe { *__v139 }) ^ __v138;
unsafe {
*__v139 = __v141;
}
__v114 = __v141;
let mut __v142: u32;
let __v143: *mut u32 = $io.wrapping_offset(0 as isize);
let __v144: u32 = unsafe { *__v143 };
let __v145: u32 = __v144 ^ SUBKEY_L!($i.wrapping_add(7), $subkey);
unsafe {
*__v143 = __v145;
}
let __v146: *mut u32 = $io.wrapping_offset(1 as isize);
let __v147: u32 = unsafe { *__v146 };
let __v148: u32 = __v147 ^ SUBKEY_R!($i.wrapping_add(7), $subkey);
unsafe {
*__v146 = __v148;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(3 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(2 as isize) >> 24i32) as u8) as i32) as usize] };
let __v149: u32 = $ir;
let __v150: u32 = __v149 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(3 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v150;
let __v151: u32 = $il;
let __v152: u32 = __v151 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(2 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v152;
let __v153: u32 = $ir;
let __v154: u32 = __v153 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(3 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v154;
let __v155: u32 = $il;
let __v156: u32 = __v155 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(2 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v156;
let __v157: u32 = $ir;
let __v158: u32 = __v157 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(3 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v158;
let __v159: u32 = $il;
let __v160: u32 = __v159 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(2 as isize) as u8) as i32) as usize] };
$il = __v160;
let __v161: u32 = $ir;
let __v162: u32 = __v161 ^ $il;
$ir = __v162;
let __v163: *mut u32 = $io.wrapping_offset(0 as isize);
let __v164: u32 = unsafe { *__v163 };
let __v165: u32 = __v164 ^ $ir;
unsafe {
*__v163 = __v165;
}
let __v166: u32 = ror32($il, 8u32) ^ $ir;
let __v167: *mut u32 = $io.wrapping_offset(1 as isize);
let __v169: u32 = (unsafe { *__v167 }) ^ __v166;
unsafe {
*__v167 = __v169;
}
__v142 = __v169;
__v = __v142;
};
}

macro_rules! ROUNDS__2 {
($i:expr, $io:expr, $subkey:expr, $ir:expr, $il:expr) => {
let mut __v: u32;
let mut __v2: u32;
let __v3: *mut u32 = $io.wrapping_offset(2 as isize);
let __v4: u32 = unsafe { *__v3 };
let __v5: u32 = __v4 ^ SUBKEY_L!($i.wrapping_add(7), $subkey);
unsafe {
*__v3 = __v5;
}
let __v6: *mut u32 = $io.wrapping_offset(3 as isize);
let __v7: u32 = unsafe { *__v6 };
let __v8: u32 = __v7 ^ SUBKEY_R!($i.wrapping_add(7), $subkey);
unsafe {
*__v6 = __v8;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(1 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(0 as isize) >> 24i32) as u8) as i32) as usize] };
let __v9: u32 = $ir;
let __v10: u32 = __v9 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(1 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v10;
let __v11: u32 = $il;
let __v12: u32 = __v11 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(0 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v12;
let __v13: u32 = $ir;
let __v14: u32 = __v13 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(1 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v14;
let __v15: u32 = $il;
let __v16: u32 = __v15 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(0 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v16;
let __v17: u32 = $ir;
let __v18: u32 = __v17 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(1 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v18;
let __v19: u32 = $il;
let __v20: u32 = __v19 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(0 as isize) as u8) as i32) as usize] };
$il = __v20;
let __v21: u32 = $ir;
let __v22: u32 = __v21 ^ $il;
$ir = __v22;
let __v23: *mut u32 = $io.wrapping_offset(2 as isize);
let __v24: u32 = unsafe { *__v23 };
let __v25: u32 = __v24 ^ $ir;
unsafe {
*__v23 = __v25;
}
let __v26: u32 = ror32($il, 8u32) ^ $ir;
let __v27: *mut u32 = $io.wrapping_offset(3 as isize);
let __v29: u32 = (unsafe { *__v27 }) ^ __v26;
unsafe {
*__v27 = __v29;
}
__v2 = __v29;
let mut __v30: u32;
let __v31: *mut u32 = $io.wrapping_offset(0 as isize);
let __v32: u32 = unsafe { *__v31 };
let __v33: u32 = __v32 ^ SUBKEY_L!($i.wrapping_add(6), $subkey);
unsafe {
*__v31 = __v33;
}
let __v34: *mut u32 = $io.wrapping_offset(1 as isize);
let __v35: u32 = unsafe { *__v34 };
let __v36: u32 = __v35 ^ SUBKEY_R!($i.wrapping_add(6), $subkey);
unsafe {
*__v34 = __v36;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(3 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(2 as isize) >> 24i32) as u8) as i32) as usize] };
let __v37: u32 = $ir;
let __v38: u32 = __v37 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(3 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v38;
let __v39: u32 = $il;
let __v40: u32 = __v39 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(2 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v40;
let __v41: u32 = $ir;
let __v42: u32 = __v41 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(3 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v42;
let __v43: u32 = $il;
let __v44: u32 = __v43 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(2 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v44;
let __v45: u32 = $ir;
let __v46: u32 = __v45 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(3 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v46;
let __v47: u32 = $il;
let __v48: u32 = __v47 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(2 as isize) as u8) as i32) as usize] };
$il = __v48;
let __v49: u32 = $ir;
let __v50: u32 = __v49 ^ $il;
$ir = __v50;
let __v51: *mut u32 = $io.wrapping_offset(0 as isize);
let __v52: u32 = unsafe { *__v51 };
let __v53: u32 = __v52 ^ $ir;
unsafe {
*__v51 = __v53;
}
let __v54: u32 = ror32($il, 8u32) ^ $ir;
let __v55: *mut u32 = $io.wrapping_offset(1 as isize);
let __v57: u32 = (unsafe { *__v55 }) ^ __v54;
unsafe {
*__v55 = __v57;
}
__v30 = __v57;
let mut __v58: u32;
let __v59: *mut u32 = $io.wrapping_offset(2 as isize);
let __v60: u32 = unsafe { *__v59 };
let __v61: u32 = __v60 ^ SUBKEY_L!($i.wrapping_add(5), $subkey);
unsafe {
*__v59 = __v61;
}
let __v62: *mut u32 = $io.wrapping_offset(3 as isize);
let __v63: u32 = unsafe { *__v62 };
let __v64: u32 = __v63 ^ SUBKEY_R!($i.wrapping_add(5), $subkey);
unsafe {
*__v62 = __v64;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(1 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(0 as isize) >> 24i32) as u8) as i32) as usize] };
let __v65: u32 = $ir;
let __v66: u32 = __v65 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(1 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v66;
let __v67: u32 = $il;
let __v68: u32 = __v67 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(0 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v68;
let __v69: u32 = $ir;
let __v70: u32 = __v69 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(1 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v70;
let __v71: u32 = $il;
let __v72: u32 = __v71 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(0 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v72;
let __v73: u32 = $ir;
let __v74: u32 = __v73 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(1 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v74;
let __v75: u32 = $il;
let __v76: u32 = __v75 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(0 as isize) as u8) as i32) as usize] };
$il = __v76;
let __v77: u32 = $ir;
let __v78: u32 = __v77 ^ $il;
$ir = __v78;
let __v79: *mut u32 = $io.wrapping_offset(2 as isize);
let __v80: u32 = unsafe { *__v79 };
let __v81: u32 = __v80 ^ $ir;
unsafe {
*__v79 = __v81;
}
let __v82: u32 = ror32($il, 8u32) ^ $ir;
let __v83: *mut u32 = $io.wrapping_offset(3 as isize);
let __v85: u32 = (unsafe { *__v83 }) ^ __v82;
unsafe {
*__v83 = __v85;
}
__v58 = __v85;
let mut __v86: u32;
let __v87: *mut u32 = $io.wrapping_offset(0 as isize);
let __v88: u32 = unsafe { *__v87 };
let __v89: u32 = __v88 ^ SUBKEY_L!($i.wrapping_add(4), $subkey);
unsafe {
*__v87 = __v89;
}
let __v90: *mut u32 = $io.wrapping_offset(1 as isize);
let __v91: u32 = unsafe { *__v90 };
let __v92: u32 = __v91 ^ SUBKEY_R!($i.wrapping_add(4), $subkey);
unsafe {
*__v90 = __v92;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(3 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(2 as isize) >> 24i32) as u8) as i32) as usize] };
let __v93: u32 = $ir;
let __v94: u32 = __v93 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(3 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v94;
let __v95: u32 = $il;
let __v96: u32 = __v95 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(2 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v96;
let __v97: u32 = $ir;
let __v98: u32 = __v97 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(3 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v98;
let __v99: u32 = $il;
let __v100: u32 = __v99 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(2 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v100;
let __v101: u32 = $ir;
let __v102: u32 = __v101 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(3 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v102;
let __v103: u32 = $il;
let __v104: u32 = __v103 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(2 as isize) as u8) as i32) as usize] };
$il = __v104;
let __v105: u32 = $ir;
let __v106: u32 = __v105 ^ $il;
$ir = __v106;
let __v107: *mut u32 = $io.wrapping_offset(0 as isize);
let __v108: u32 = unsafe { *__v107 };
let __v109: u32 = __v108 ^ $ir;
unsafe {
*__v107 = __v109;
}
let __v110: u32 = ror32($il, 8u32) ^ $ir;
let __v111: *mut u32 = $io.wrapping_offset(1 as isize);
let __v113: u32 = (unsafe { *__v111 }) ^ __v110;
unsafe {
*__v111 = __v113;
}
__v86 = __v113;
let mut __v114: u32;
let __v115: *mut u32 = $io.wrapping_offset(2 as isize);
let __v116: u32 = unsafe { *__v115 };
let __v117: u32 = __v116 ^ SUBKEY_L!($i.wrapping_add(3), $subkey);
unsafe {
*__v115 = __v117;
}
let __v118: *mut u32 = $io.wrapping_offset(3 as isize);
let __v119: u32 = unsafe { *__v118 };
let __v120: u32 = __v119 ^ SUBKEY_R!($i.wrapping_add(3), $subkey);
unsafe {
*__v118 = __v120;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(1 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(0 as isize) >> 24i32) as u8) as i32) as usize] };
let __v121: u32 = $ir;
let __v122: u32 = __v121 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(1 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v122;
let __v123: u32 = $il;
let __v124: u32 = __v123 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(0 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v124;
let __v125: u32 = $ir;
let __v126: u32 = __v125 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(1 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v126;
let __v127: u32 = $il;
let __v128: u32 = __v127 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(0 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v128;
let __v129: u32 = $ir;
let __v130: u32 = __v129 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(1 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v130;
let __v131: u32 = $il;
let __v132: u32 = __v131 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(0 as isize) as u8) as i32) as usize] };
$il = __v132;
let __v133: u32 = $ir;
let __v134: u32 = __v133 ^ $il;
$ir = __v134;
let __v135: *mut u32 = $io.wrapping_offset(2 as isize);
let __v136: u32 = unsafe { *__v135 };
let __v137: u32 = __v136 ^ $ir;
unsafe {
*__v135 = __v137;
}
let __v138: u32 = ror32($il, 8u32) ^ $ir;
let __v139: *mut u32 = $io.wrapping_offset(3 as isize);
let __v141: u32 = (unsafe { *__v139 }) ^ __v138;
unsafe {
*__v139 = __v141;
}
__v114 = __v141;
let mut __v142: u32;
let __v143: *mut u32 = $io.wrapping_offset(0 as isize);
let __v144: u32 = unsafe { *__v143 };
let __v145: u32 = __v144 ^ SUBKEY_L!($i.wrapping_add(2), $subkey);
unsafe {
*__v143 = __v145;
}
let __v146: *mut u32 = $io.wrapping_offset(1 as isize);
let __v147: u32 = unsafe { *__v146 };
let __v148: u32 = __v147 ^ SUBKEY_R!($i.wrapping_add(2), $subkey);
unsafe {
*__v146 = __v148;
}
$ir = unsafe { camellia_sp1110.0[((*$io.wrapping_offset(3 as isize) as u8) as i32) as usize] };
$il = unsafe { camellia_sp1110.0[(((*$io.wrapping_offset(2 as isize) >> 24i32) as u8) as i32) as usize] };
let __v149: u32 = $ir;
let __v150: u32 = __v149 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(3 as isize) >> 24i32) as u8) as i32) as usize] };
$ir = __v150;
let __v151: u32 = $il;
let __v152: u32 = __v151 ^ unsafe { camellia_sp0222.0[(((*$io.wrapping_offset(2 as isize) >> 16i32) as u8) as i32) as usize] };
$il = __v152;
let __v153: u32 = $ir;
let __v154: u32 = __v153 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(3 as isize) >> 16i32) as u8) as i32) as usize] };
$ir = __v154;
let __v155: u32 = $il;
let __v156: u32 = __v155 ^ unsafe { camellia_sp3033.0[(((*$io.wrapping_offset(2 as isize) >> 8i32) as u8) as i32) as usize] };
$il = __v156;
let __v157: u32 = $ir;
let __v158: u32 = __v157 ^ unsafe { camellia_sp4404.0[(((*$io.wrapping_offset(3 as isize) >> 8i32) as u8) as i32) as usize] };
$ir = __v158;
let __v159: u32 = $il;
let __v160: u32 = __v159 ^ unsafe { camellia_sp4404.0[((*$io.wrapping_offset(2 as isize) as u8) as i32) as usize] };
$il = __v160;
let __v161: u32 = $ir;
let __v162: u32 = __v161 ^ $il;
$ir = __v162;
let __v163: *mut u32 = $io.wrapping_offset(0 as isize);
let __v164: u32 = unsafe { *__v163 };
let __v165: u32 = __v164 ^ $ir;
unsafe {
*__v163 = __v165;
}
let __v166: u32 = ror32($il, 8u32) ^ $ir;
let __v167: *mut u32 = $io.wrapping_offset(1 as isize);
let __v169: u32 = (unsafe { *__v167 }) ^ __v166;
unsafe {
*__v167 = __v169;
}
__v142 = __v169;
__v = __v142;
};
}

macro_rules! SUBKEY_L {
($INDEX:expr, $subkey:expr) => {
unsafe { *$subkey.wrapping_offset($INDEX.wrapping_mul(2) as isize) }
};
}

macro_rules! SUBKEY_L__2 {
($INDEX:expr, $subkey:expr) => {
unsafe { *$subkey.wrapping_offset($INDEX.wrapping_mul(2) as isize) }
};
}

macro_rules! SUBKEY_R {
($INDEX:expr, $subkey:expr) => {
unsafe { *$subkey.wrapping_offset($INDEX.wrapping_mul(2).wrapping_add(1) as isize) }
};
}

macro_rules! SUBKEY_R__2 {
($INDEX:expr, $subkey:expr) => {
unsafe { *$subkey.wrapping_offset($INDEX.wrapping_mul(2).wrapping_add(1) as isize) }
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t__2 {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u32>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
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

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t {
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = $val;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
};
}

macro_rules! __swab32 {
($x:expr) => {
unsafe { __slate_intrinsic_e94e9cf815168b51(($x) as u32) }
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

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __bad_size_call_parameter();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __compiletime_assert_57();
    fn crypto_register_alg(arg0: *mut crypto_alg) -> i32;
    fn crypto_unregister_alg(arg0: *mut crypto_alg);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_368();
    fn __compiletime_assert_393();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bswap.i32"]
    fn __slate_intrinsic_e94e9cf815168b51(arg0: u32) -> u32;
}

// generic data direction definitions
// The type of an index into the pagecache.
/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
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


fn get_unaligned_be32(p: *const ()) -> u32 {
__swab32!(__get_unaligned_t__2!(u32, p))
}

fn put_unaligned_be32(val: u32, p: *mut ()) {
__put_unaligned_t!(u32, __swab32!(val), p);
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
struct lock_class_key {
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits91U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits92U0,
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn rol32(word: u32, shift: u32) -> u32 {
word << (shift & 31) | word >> (shift.wrapping_neg() & 31)
}

fn ror32(word: u32, shift: u32) -> u32 {
word >> (shift & 31) | word << (shift.wrapping_neg() & 31)
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
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_57() };
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
    __slate_bits_0: __slate_bits::__SlateBits102U0,
    pkru: u32,
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
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct seqcount {
    sequence: u32,
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
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
struct kref {
    refcount: refcount_struct,
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
}


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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct seq_file {
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
    __slate_bits_0: __slate_bits::__SlateBits260U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct gendisk {
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
struct vm_struct {
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
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
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
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits374U0,
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Set if the algorithm supports virtual addresses.
// Miscellaneous stuff.
#[repr(C)]
#[derive(Clone, Copy)]
struct cipher_alg {
    cia_min_keysize: u32,
    cia_max_keysize: u32,
    cia_setkey: Option<unsafe extern "C" fn(*mut crypto_tfm, *const u8, u32) -> i32>,
    cia_encrypt: Option<unsafe extern "C" fn(*mut crypto_tfm, *mut u8, *const u8)>,
    cia_decrypt: Option<unsafe extern "C" fn(*mut crypto_tfm, *mut u8, *const u8)>,
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
struct dev_pagemap {
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
struct crypto_alg {
    cra_list: list_head,
    cra_users: list_head,
    cra_flags: u32,
    cra_blocksize: u32,
    cra_ctxsize: u32,
    cra_alignmask: u32,
    cra_reqsize: u32,
    cra_priority: i32,
    cra_refcnt: refcount_struct,
    cra_name: [u8; 128],
    cra_driver_name: [u8; 128],
    cra_type: *const crypto_type,
    cra_u: __SlateRecord_crypto_h_350,
    cra_init: Option<unsafe extern "C" fn(*mut crypto_tfm) -> i32>,
    cra_exit: Option<unsafe extern "C" fn(*mut crypto_tfm)>,
    cra_destroy: Option<unsafe extern "C" fn(*mut crypto_alg)>,
    cra_module: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct crypto_tfm {
    crt_flags: u32,
    node: i32,
    fb: *mut crypto_tfm,
    exit: Option<unsafe extern "C" fn(*mut crypto_tfm)>,
    __crt_alg: *mut crypto_alg,
    __crt_ctx: [*mut (); 0],
}


fn crypto_tfm_ctx(tfm: *mut crypto_tfm) -> *mut () {
(unsafe { core::ptr::addr_of_mut!((*tfm).__crt_ctx) as *mut *mut () }) as *mut ()
}

// When an algorithm uses another algorithm (e.g., if it's an instance of a
// template), these are the flags that should always be set on the "outer"
// algorithm if any "inner" algorithm has them set.
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
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
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
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
struct crypto_type {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_crypto_h_350 {
    cipher: cipher_alg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


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
struct folio_queue {
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
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
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
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
    __slate_bits_0: __slate_bits::__SlateBits263U0,
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
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_368() };
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
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


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
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


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct poll_table_struct {
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


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_393() };
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
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
struct io_uring_cmd {
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
struct kernfs_root {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
    __slate_bits_0: __slate_bits::__SlateBits588U0,
    vm_ops: *const vm_operations_struct,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits532U0,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits260U0 {
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
    pub struct __SlateBits263U0 {
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
    pub struct __SlateBits92U0 {
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
    pub struct __SlateBits102U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits588U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits532U0 {
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
    pub struct __SlateBits91U0 {
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
    pub struct __SlateBits374U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
// module_exit() - driver exit entry point
// @x: function to be run when driver is removed
//
// module_exit() will wrap the driver clean-up code
// with cleanup_module() when used with rmmod when
// the driver is a module.  If the driver is statically
// compiled into the kernel, module_exit() has no effect.
// There can only be one per module.
// The following license idents are currently accepted as indicating free
// software modules
//
// 	"GPL"				[GNU Public License v2]
// 	"GPL v2"			[GNU Public License v2]
// 	"GPL and additional rights"	[GNU Public License v2 rights and more]
// 	"Dual BSD/GPL"			[GNU Public License v2
// 					 or BSD license choice]
// 	"Dual MIT/GPL"			[GNU Public License v2
// 					 or MIT license choice]
// 	"Dual MPL/GPL"			[GNU Public License v2
// 					 or Mozilla license choice]
//
// The following other idents are available
//
// 	"Proprietary"			[Non free products]
//
// Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
// merely stating that the module is licensed under the GPL v2, but are not
// telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
// are two variants is a historic and failed attempt to convey more
// information in the MODULE_LICENSE string. For module loading the
// "only/or later" distinction is completely irrelevant and does neither
// replace the proper license identifiers in the corresponding source file
// nor amends them in any way. The sole purpose is to make the
// 'Proprietary' flagging work and to refuse to bind symbols which are
// exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
//
// In the same way "BSD" is not a clear license information. It merely
// states, that the module is licensed under one of the compatible BSD
// license variants. The detailed and correct license information is again
// to be found in the corresponding source files.
//
// There are dual licensed components, but when running with Linux it is the
// GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
// is a GPL combined work.
//
// This exists for several reasons
// 1.	So modinfo can show license info for users wanting to vet their setup
// 	is free
// 2.	So the community can ignore bug reports including proprietary modules
// 3.	So vendors can do likewise based on their own policies
// What your module does.
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


// Algorithm Specification
//  https://info.isl.ntt.co.jp/crypt/eng/camellia/specifications.html
static camellia_sp1110: __SlateAlign16<[u32; 256]> = __SlateAlign16([1886416896u32, 2189591040u32, 741092352u32, 3974949888u32, 3014898432u32, 656877312u32, 3233857536u32, 3857048832u32, 3840205824u32, 2240120064u32, 1465341696u32, 892679424u32, 3941263872u32, 202116096u32, 2930683392u32, 1094795520u32, 589505280u32, 4025478912u32, 1802201856u32, 2475922176u32, 1162167552u32, 421075200u32, 2779096320u32, 555819264u32, 3991792896u32, 235802112u32, 1330597632u32, 1313754624u32, 488447232u32, 1701143808u32, 2459079168u32, 3183328512u32, 2256963072u32, 3099113472u32, 2947526400u32, 2408550144u32, 2088532992u32, 3958106880u32, 522133248u32, 3469659648u32, 1044266496u32, 808464384u32, 3705461760u32, 1600085760u32, 1583242752u32, 3318072576u32, 185273088u32, 437918208u32, 2795939328u32, 3789676800u32, 960051456u32, 3402287616u32, 3587560704u32, 1195853568u32, 1566399744u32, 1027423488u32, 3654932736u32, 16843008u32, 1515870720u32, 3604403712u32, 1364283648u32, 1448498688u32, 1819044864u32, 1296911616u32, 2341178112u32, 218959104u32, 2593823232u32, 1717986816u32, 4227595008u32, 3435973632u32, 2964369408u32, 757935360u32, 1953788928u32, 303174144u32, 724249344u32, 538976256u32, 4042321920u32, 2981212416u32, 2223277056u32, 2576980224u32, 3755990784u32, 1280068608u32, 3419130624u32, 3267543552u32, 875836416u32, 2122219008u32, 1987474944u32, 84215040u32, 1835887872u32, 3082270464u32, 2846468352u32, 825307392u32, 3520188672u32, 387389184u32, 67372032u32, 3621246720u32, 336860160u32, 1482184704u32, 976894464u32, 1633771776u32, 3739147776u32, 454761216u32, 286331136u32, 471604224u32, 842150400u32, 252645120u32, 2627509248u32, 370546176u32, 1397969664u32, 404232192u32, 4076007936u32, 572662272u32, 4278124032u32, 1145324544u32, 3486502656u32, 2998055424u32, 3284386560u32, 3048584448u32, 2054846976u32, 2442236160u32, 606348288u32, 134744064u32, 3907577856u32, 2829625344u32, 1616928768u32, 4244438016u32, 1768515840u32, 1347440640u32, 2863311360u32, 3503345664u32, 2694881280u32, 2105376000u32, 2711724288u32, 2307492096u32, 1650614784u32, 2543294208u32, 1414812672u32, 1532713728u32, 505290240u32, 2509608192u32, 3772833792u32, 4294967040u32, 1684300800u32, 3537031680u32, 269488128u32, 3301229568u32, 0u32, 1212696576u32, 2745410304u32, 4160222976u32, 1970631936u32, 3688618752u32, 2324335104u32, 50529024u32, 3873891840u32, 3671775744u32, 151587072u32, 1061109504u32, 3722304768u32, 2492765184u32, 2273806080u32, 1549556736u32, 2206434048u32, 33686016u32, 3452816640u32, 1246382592u32, 2425393152u32, 858993408u32, 1936945920u32, 1734829824u32, 4143379968u32, 4092850944u32, 2644352256u32, 2139062016u32, 3217014528u32, 3806519808u32, 1381126656u32, 2610666240u32, 3638089728u32, 640034304u32, 3368601600u32, 926365440u32, 3334915584u32, 993737472u32, 2172748032u32, 2526451200u32, 1869573888u32, 1263225600u32, 320017152u32, 3200171520u32, 1667457792u32, 774778368u32, 3924420864u32, 2038003968u32, 2812782336u32, 2358021120u32, 2678038272u32, 1852730880u32, 3166485504u32, 2391707136u32, 690563328u32, 4126536960u32, 4193908992u32, 3065427456u32, 791621376u32, 4261281024u32, 3031741440u32, 1499027712u32, 2021160960u32, 2560137216u32, 101058048u32, 1785358848u32, 3890734848u32, 1179010560u32, 1903259904u32, 3132799488u32, 3570717696u32, 623191296u32, 2880154368u32, 1111638528u32, 2290649088u32, 2728567296u32, 2374864128u32, 4210752000u32, 1920102912u32, 117901056u32, 3115956480u32, 1431655680u32, 4177065984u32, 4008635904u32, 2896997376u32, 168430080u32, 909522432u32, 1229539584u32, 707406336u32, 1751672832u32, 1010580480u32, 943208448u32, 4059164928u32, 2762253312u32, 1077952512u32, 673720320u32, 3553874688u32, 2071689984u32, 3149642496u32, 3385444608u32, 1128481536u32, 3250700544u32, 353703168u32, 3823362816u32, 2913840384u32, 4109693952u32, 2004317952u32, 3351758592u32, 2155905024u32, 2661195264u32]);

core::arch::global_asm!(".section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_camellia_generic__441_1067_camellia_init6:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n", sym camellia_init, options(att_syntax));

static camellia_sp0222: __SlateAlign16<[u32; 256]> = __SlateAlign16([14737632u32, 328965u32, 5789784u32, 14277081u32, 6776679u32, 5131854u32, 8487297u32, 13355979u32, 13224393u32, 723723u32, 11447982u32, 6974058u32, 14013909u32, 1579032u32, 6118749u32, 8553090u32, 4605510u32, 14671839u32, 14079702u32, 2565927u32, 9079434u32, 3289650u32, 4934475u32, 4342338u32, 14408667u32, 1842204u32, 10395294u32, 10263708u32, 3815994u32, 13290186u32, 2434341u32, 8092539u32, 855309u32, 7434609u32, 6250335u32, 2039583u32, 16316664u32, 14145495u32, 4079166u32, 10329501u32, 8158332u32, 6316128u32, 12171705u32, 12500670u32, 12369084u32, 9145227u32, 1447446u32, 3421236u32, 5066061u32, 12829635u32, 7500402u32, 9803157u32, 11250603u32, 9342606u32, 12237498u32, 8026746u32, 11776947u32, 131586u32, 11842740u32, 11382189u32, 10658466u32, 11316396u32, 14211288u32, 10132122u32, 1513239u32, 1710618u32, 3487029u32, 13421772u32, 16250871u32, 10066329u32, 6381921u32, 5921370u32, 15263976u32, 2368548u32, 5658198u32, 4210752u32, 14803425u32, 6513507u32, 592137u32, 3355443u32, 12566463u32, 10000536u32, 9934743u32, 8750469u32, 6842472u32, 16579836u32, 15527148u32, 657930u32, 14342874u32, 7303023u32, 5460819u32, 6447714u32, 10724259u32, 3026478u32, 526344u32, 11513775u32, 2631720u32, 11579568u32, 7631988u32, 12763842u32, 12434877u32, 3552822u32, 2236962u32, 3684408u32, 6579300u32, 1973790u32, 3750201u32, 2894892u32, 10921638u32, 3158064u32, 15066597u32, 4473924u32, 16645629u32, 8947848u32, 10461087u32, 6645093u32, 8882055u32, 7039851u32, 16053492u32, 2302755u32, 4737096u32, 1052688u32, 13750737u32, 5329233u32, 12632256u32, 16382457u32, 13816530u32, 10526880u32, 5592405u32, 10592673u32, 4276545u32, 16448250u32, 4408131u32, 1250067u32, 12895428u32, 3092271u32, 11053224u32, 11974326u32, 3947580u32, 2829099u32, 12698049u32, 16777215u32, 13158600u32, 10855845u32, 2105376u32, 9013641u32, 0u32, 9474192u32, 4671303u32, 15724527u32, 15395562u32, 12040119u32, 1381653u32, 394758u32, 13487565u32, 11908533u32, 1184274u32, 8289918u32, 12303291u32, 2697513u32, 986895u32, 12105912u32, 460551u32, 263172u32, 10197915u32, 9737364u32, 2171169u32, 6710886u32, 15132390u32, 13553358u32, 15592941u32, 15198183u32, 3881787u32, 16711422u32, 8355711u32, 12961221u32, 10790052u32, 3618615u32, 11645361u32, 5000268u32, 9539985u32, 7237230u32, 9276813u32, 7763574u32, 197379u32, 2960685u32, 14606046u32, 9868950u32, 2500134u32, 8224125u32, 13027014u32, 6052956u32, 13882323u32, 15921906u32, 5197647u32, 1644825u32, 4144959u32, 14474460u32, 7960953u32, 1907997u32, 5395026u32, 15461355u32, 15987699u32, 7171437u32, 6184542u32, 16514043u32, 6908265u32, 11711154u32, 15790320u32, 3223857u32, 789516u32, 13948116u32, 13619151u32, 9211020u32, 14869218u32, 7697781u32, 11119017u32, 4868682u32, 5723991u32, 8684676u32, 1118481u32, 4539717u32, 1776411u32, 16119285u32, 15000804u32, 921102u32, 7566195u32, 11184810u32, 15856113u32, 14540253u32, 5855577u32, 1315860u32, 7105644u32, 9605778u32, 5526612u32, 13684944u32, 7895160u32, 7368816u32, 14935011u32, 4802889u32, 8421504u32, 5263440u32, 10987431u32, 16185078u32, 7829367u32, 9671571u32, 8816262u32, 8618883u32, 2763306u32, 13092807u32, 5987163u32, 15329769u32, 15658734u32, 9408399u32, 65793u32, 4013373u32]);

static camellia_sp3033: __SlateAlign16<[u32; 256]> = __SlateAlign16([939538488u32, 1090535745u32, 369104406u32, 1979741814u32, 3640711641u32, 2466288531u32, 1610637408u32, 4060148466u32, 1912631922u32, 3254829762u32, 2868947883u32, 2583730842u32, 1962964341u32, 100664838u32, 1459640151u32, 2684395680u32, 2432733585u32, 4144035831u32, 3036722613u32, 3372272073u32, 2717950626u32, 2348846220u32, 3523269330u32, 2415956112u32, 4127258358u32, 117442311u32, 2801837991u32, 654321447u32, 2382401166u32, 2986390194u32, 1224755529u32, 3724599006u32, 1124090691u32, 1543527516u32, 3607156695u32, 3338717127u32, 1040203326u32, 4110480885u32, 2399178639u32, 1728079719u32, 520101663u32, 402659352u32, 1845522030u32, 2936057775u32, 788541231u32, 3791708898u32, 2231403909u32, 218107149u32, 1392530259u32, 4026593520u32, 2617285788u32, 1694524773u32, 3925928682u32, 2734728099u32, 2919280302u32, 2650840734u32, 3959483628u32, 2147516544u32, 754986285u32, 1795189611u32, 2818615464u32, 721431339u32, 905983542u32, 2785060518u32, 3305162181u32, 2248181382u32, 1291865421u32, 855651123u32, 4244700669u32, 1711302246u32, 1476417624u32, 2516620950u32, 973093434u32, 150997257u32, 2499843477u32, 268439568u32, 2013296760u32, 3623934168u32, 1107313218u32, 3422604492u32, 4009816047u32, 637543974u32, 3842041317u32, 1627414881u32, 436214298u32, 1056980799u32, 989870907u32, 2181071490u32, 3053500086u32, 3674266587u32, 3556824276u32, 2550175896u32, 3892373736u32, 2332068747u32, 33554946u32, 3942706155u32, 167774730u32, 738208812u32, 486546717u32, 2952835248u32, 1862299503u32, 2365623693u32, 2281736328u32, 234884622u32, 419436825u32, 2264958855u32, 1308642894u32, 184552203u32, 2835392937u32, 201329676u32, 2030074233u32, 285217041u32, 2130739071u32, 570434082u32, 3875596263u32, 1493195097u32, 3774931425u32, 3657489114u32, 1023425853u32, 3355494600u32, 301994514u32, 67109892u32, 1946186868u32, 1409307732u32, 805318704u32, 2113961598u32, 3019945140u32, 671098920u32, 1426085205u32, 1744857192u32, 1342197840u32, 3187719870u32, 3489714384u32, 3288384708u32, 822096177u32, 3405827019u32, 704653866u32, 2902502829u32, 251662095u32, 3389049546u32, 1879076976u32, 4278255615u32, 838873650u32, 1761634665u32, 134219784u32, 1644192354u32, 0u32, 603989028u32, 3506491857u32, 4211145723u32, 3120609978u32, 3976261101u32, 1157645637u32, 2164294017u32, 1929409395u32, 1828744557u32, 2214626436u32, 2667618207u32, 3993038574u32, 1241533002u32, 3271607235u32, 771763758u32, 3238052289u32, 16777473u32, 3858818790u32, 620766501u32, 1207978056u32, 2566953369u32, 3103832505u32, 3003167667u32, 2063629179u32, 4177590777u32, 3456159438u32, 3204497343u32, 3741376479u32, 1895854449u32, 687876393u32, 3439381965u32, 1811967084u32, 318771987u32, 1677747300u32, 2600508315u32, 1660969827u32, 2634063261u32, 3221274816u32, 1258310475u32, 3070277559u32, 2768283045u32, 2298513801u32, 1593859935u32, 2969612721u32, 385881879u32, 4093703412u32, 3154164924u32, 3540046803u32, 1174423110u32, 3472936911u32, 922761015u32, 1577082462u32, 1191200583u32, 2483066004u32, 4194368250u32, 4227923196u32, 1526750043u32, 2533398423u32, 4261478142u32, 1509972570u32, 2885725356u32, 1006648380u32, 1275087948u32, 50332419u32, 889206069u32, 4076925939u32, 587211555u32, 3087055032u32, 1560304989u32, 1778412138u32, 2449511058u32, 3573601749u32, 553656609u32, 1140868164u32, 1358975313u32, 3321939654u32, 2097184125u32, 956315961u32, 2197848963u32, 3691044060u32, 2852170410u32, 2080406652u32, 1996519287u32, 1442862678u32, 83887365u32, 452991771u32, 2751505572u32, 352326933u32, 872428596u32, 503324190u32, 469769244u32, 4160813304u32, 1375752786u32, 536879136u32, 335549460u32, 3909151209u32, 3170942397u32, 3707821533u32, 3825263844u32, 2701173153u32, 3758153952u32, 2315291274u32, 4043370993u32, 3590379222u32, 2046851706u32, 3137387451u32, 3808486371u32, 1073758272u32, 1325420367u32]);

static camellia_sp4404: __SlateAlign16<[u32; 256]> = __SlateAlign16([1886388336u32, 741081132u32, 3014852787u32, 3233808576u32, 3840147684u32, 1465319511u32, 3941204202u32, 2930639022u32, 589496355u32, 1802174571u32, 1162149957u32, 2779054245u32, 3991732461u32, 1330577487u32, 488439837u32, 2459041938u32, 2256928902u32, 2947481775u32, 2088501372u32, 522125343u32, 1044250686u32, 3705405660u32, 1583218782u32, 185270283u32, 2795896998u32, 960036921u32, 3587506389u32, 1566376029u32, 3654877401u32, 1515847770u32, 1364262993u32, 1819017324u32, 2341142667u32, 2593783962u32, 4227531003u32, 2964324528u32, 1953759348u32, 724238379u32, 4042260720u32, 2223243396u32, 3755933919u32, 3419078859u32, 875823156u32, 1987444854u32, 1835860077u32, 2846425257u32, 3520135377u32, 67371012u32, 336855060u32, 976879674u32, 3739091166u32, 286326801u32, 842137650u32, 2627469468u32, 1397948499u32, 4075946226u32, 4278059262u32, 3486449871u32, 3284336835u32, 2054815866u32, 606339108u32, 3907518696u32, 1616904288u32, 1768489065u32, 2863268010u32, 2694840480u32, 2711683233u32, 1650589794u32, 1414791252u32, 505282590u32, 3772776672u32, 1684275300u32, 269484048u32, 0u32, 2745368739u32, 1970602101u32, 2324299914u32, 3873833190u32, 151584777u32, 3722248413u32, 2273771655u32, 2206400643u32, 3452764365u32, 2425356432u32, 1936916595u32, 4143317238u32, 2644312221u32, 3216965823u32, 1381105746u32, 3638034648u32, 3368550600u32, 3334865094u32, 2172715137u32, 1869545583u32, 320012307u32, 1667432547u32, 3924361449u32, 2812739751u32, 2677997727u32, 3166437564u32, 690552873u32, 4193845497u32, 791609391u32, 3031695540u32, 2021130360u32, 101056518u32, 3890675943u32, 1903231089u32, 3570663636u32, 2880110763u32, 2290614408u32, 2374828173u32, 1920073842u32, 3115909305u32, 4177002744u32, 2896953516u32, 909508662u32, 707395626u32, 1010565180u32, 4059103473u32, 1077936192u32, 3553820883u32, 3149594811u32, 1128464451u32, 353697813u32, 2913796269u32, 2004287607u32, 2155872384u32, 2189557890u32, 3974889708u32, 656867367u32, 3856990437u32, 2240086149u32, 892665909u32, 202113036u32, 1094778945u32, 4025417967u32, 2475884691u32, 421068825u32, 555810849u32, 235798542u32, 1313734734u32, 1701118053u32, 3183280317u32, 3099066552u32, 2408513679u32, 3958046955u32, 3469607118u32, 808452144u32, 1600061535u32, 3318022341u32, 437911578u32, 3789619425u32, 3402236106u32, 1195835463u32, 1027407933u32, 16842753u32, 3604349142u32, 1448476758u32, 1296891981u32, 218955789u32, 1717960806u32, 3435921612u32, 757923885u32, 303169554u32, 538968096u32, 2981167281u32, 2576941209u32, 1280049228u32, 3267494082u32, 2122186878u32, 84213765u32, 3082223799u32, 825294897u32, 387383319u32, 3621191895u32, 1482162264u32, 1633747041u32, 454754331u32, 471597084u32, 252641295u32, 370540566u32, 404226072u32, 572653602u32, 1145307204u32, 2998010034u32, 3048538293u32, 2442199185u32, 134742024u32, 2829582504u32, 4244373756u32, 1347420240u32, 3503292624u32, 2105344125u32, 2307457161u32, 2543255703u32, 1532690523u32, 2509570197u32, 4294902015u32, 3536978130u32, 3301179588u32, 1212678216u32, 4160159991u32, 3688562907u32, 50528259u32, 3671720154u32, 1061093439u32, 2492727444u32, 1549533276u32, 33685506u32, 1246363722u32, 858980403u32, 1734803559u32, 4092788979u32, 2139029631u32, 3806462178u32, 2610626715u32, 640024614u32, 926351415u32, 993722427u32, 2526412950u32, 1263206475u32, 3200123070u32, 774766638u32, 2037973113u32, 2357985420u32, 1852702830u32, 2391670926u32, 4126474485u32, 3065381046u32, 4261216509u32, 1499005017u32, 2560098456u32, 1785331818u32, 1178992710u32, 3132752058u32, 623181861u32, 1111621698u32, 2728525986u32, 4210688250u32, 117899271u32, 1431634005u32, 4008575214u32, 168427530u32, 1229520969u32, 1751646312u32, 943194168u32, 2762211492u32, 673710120u32, 2071658619u32, 3385393353u32, 3250651329u32, 3823304931u32, 4109631732u32, 3351707847u32, 2661154974u32]);

// NB: L and R below stand for 'left' and 'right' as in written numbers.
// That is, in (xxxL,xxxR) pair xxxL holds most significant digits,
// _not_ least significant ones!
// key constants
fn camellia_setup_tail(subkey: *mut u32, subL: *mut u32, subR: *mut u32, max: i32) {
let mut dw: u32 = 0;
let mut tl: u32 = 0;
let mut tr: u32 = 0;
let mut kw4l: u32 = 0;
let mut kw4r: u32 = 0;
// absorb kw2 to other subkeys
//
// round 2
let __v0: *mut u32 = subL.wrapping_offset(3 as isize);
unsafe {
*__v0 = *__v0 ^ *subL.wrapping_offset(1 as isize);
}
let __v3: *mut u32 = subR.wrapping_offset(3 as isize);
unsafe {
*__v3 = *__v3 ^ *subR.wrapping_offset(1 as isize);
}
// round 4
let __v6: *mut u32 = subL.wrapping_offset(5 as isize);
unsafe {
*__v6 = *__v6 ^ *subL.wrapping_offset(1 as isize);
}
let __v9: *mut u32 = subR.wrapping_offset(5 as isize);
unsafe {
*__v9 = *__v9 ^ *subR.wrapping_offset(1 as isize);
}
// round 6
let __v12: *mut u32 = subL.wrapping_offset(7 as isize);
unsafe {
*__v12 = *__v12 ^ *subL.wrapping_offset(1 as isize);
}
let __v15: *mut u32 = subR.wrapping_offset(7 as isize);
unsafe {
*__v15 = *__v15 ^ *subR.wrapping_offset(1 as isize);
}
let __v18: *mut u32 = subL.wrapping_offset(1 as isize);
let __v20: u32 = (unsafe { *__v18 }) ^ (unsafe { *subR.wrapping_offset(1 as isize) }) & !unsafe { *subR.wrapping_offset(9 as isize) };
unsafe {
*__v18 = __v20;
}
dw = (unsafe { *subL.wrapping_offset(1 as isize) }) & unsafe { *subL.wrapping_offset(9 as isize) };
let __v21: u32 = rol32(dw, 1u32);
let __v22: *mut u32 = subR.wrapping_offset(1 as isize);
unsafe {
*__v22 = *__v22 ^ __v21;
}
// modified for FLinv(kl2)
// round 8
let __v25: *mut u32 = subL.wrapping_offset(11 as isize);
unsafe {
*__v25 = *__v25 ^ *subL.wrapping_offset(1 as isize);
}
let __v28: *mut u32 = subR.wrapping_offset(11 as isize);
unsafe {
*__v28 = *__v28 ^ *subR.wrapping_offset(1 as isize);
}
// round 10
let __v31: *mut u32 = subL.wrapping_offset(13 as isize);
unsafe {
*__v31 = *__v31 ^ *subL.wrapping_offset(1 as isize);
}
let __v34: *mut u32 = subR.wrapping_offset(13 as isize);
unsafe {
*__v34 = *__v34 ^ *subR.wrapping_offset(1 as isize);
}
// round 12
let __v37: *mut u32 = subL.wrapping_offset(15 as isize);
unsafe {
*__v37 = *__v37 ^ *subL.wrapping_offset(1 as isize);
}
let __v40: *mut u32 = subR.wrapping_offset(15 as isize);
unsafe {
*__v40 = *__v40 ^ *subR.wrapping_offset(1 as isize);
}
let __v43: *mut u32 = subL.wrapping_offset(1 as isize);
let __v45: u32 = (unsafe { *__v43 }) ^ (unsafe { *subR.wrapping_offset(1 as isize) }) & !unsafe { *subR.wrapping_offset(17 as isize) };
unsafe {
*__v43 = __v45;
}
dw = (unsafe { *subL.wrapping_offset(1 as isize) }) & unsafe { *subL.wrapping_offset(17 as isize) };
let __v46: u32 = rol32(dw, 1u32);
let __v47: *mut u32 = subR.wrapping_offset(1 as isize);
unsafe {
*__v47 = *__v47 ^ __v46;
}
// modified for FLinv(kl4)
// round 14
let __v50: *mut u32 = subL.wrapping_offset(19 as isize);
unsafe {
*__v50 = *__v50 ^ *subL.wrapping_offset(1 as isize);
}
let __v53: *mut u32 = subR.wrapping_offset(19 as isize);
unsafe {
*__v53 = *__v53 ^ *subR.wrapping_offset(1 as isize);
}
// round 16
let __v56: *mut u32 = subL.wrapping_offset(21 as isize);
unsafe {
*__v56 = *__v56 ^ *subL.wrapping_offset(1 as isize);
}
let __v59: *mut u32 = subR.wrapping_offset(21 as isize);
unsafe {
*__v59 = *__v59 ^ *subR.wrapping_offset(1 as isize);
}
// round 18
let __v62: *mut u32 = subL.wrapping_offset(23 as isize);
unsafe {
*__v62 = *__v62 ^ *subL.wrapping_offset(1 as isize);
}
let __v65: *mut u32 = subR.wrapping_offset(23 as isize);
unsafe {
*__v65 = *__v65 ^ *subR.wrapping_offset(1 as isize);
}
if max == 24 {
// kw3
let __v68: *mut u32 = subL.wrapping_offset(24 as isize);
unsafe {
*__v68 = *__v68 ^ *subL.wrapping_offset(1 as isize);
}
let __v71: *mut u32 = subR.wrapping_offset(24 as isize);
unsafe {
*__v71 = *__v71 ^ *subR.wrapping_offset(1 as isize);
}
// absorb kw4 to other subkeys
kw4l = unsafe { *subL.wrapping_offset(25 as isize) };
kw4r = unsafe { *subR.wrapping_offset(25 as isize) };
} else {
let __v74: *mut u32 = subL.wrapping_offset(1 as isize);
let __v76: u32 = (unsafe { *__v74 }) ^ (unsafe { *subR.wrapping_offset(1 as isize) }) & !unsafe { *subR.wrapping_offset(25 as isize) };
unsafe {
*__v74 = __v76;
}
dw = (unsafe { *subL.wrapping_offset(1 as isize) }) & unsafe { *subL.wrapping_offset(25 as isize) };
let __v77: u32 = rol32(dw, 1u32);
let __v78: *mut u32 = subR.wrapping_offset(1 as isize);
unsafe {
*__v78 = *__v78 ^ __v77;
}
// modified for FLinv(kl6)
// round 20
let __v81: *mut u32 = subL.wrapping_offset(27 as isize);
unsafe {
*__v81 = *__v81 ^ *subL.wrapping_offset(1 as isize);
}
let __v84: *mut u32 = subR.wrapping_offset(27 as isize);
unsafe {
*__v84 = *__v84 ^ *subR.wrapping_offset(1 as isize);
}
// round 22
let __v87: *mut u32 = subL.wrapping_offset(29 as isize);
unsafe {
*__v87 = *__v87 ^ *subL.wrapping_offset(1 as isize);
}
let __v90: *mut u32 = subR.wrapping_offset(29 as isize);
unsafe {
*__v90 = *__v90 ^ *subR.wrapping_offset(1 as isize);
}
// round 24
let __v93: *mut u32 = subL.wrapping_offset(31 as isize);
unsafe {
*__v93 = *__v93 ^ *subL.wrapping_offset(1 as isize);
}
let __v96: *mut u32 = subR.wrapping_offset(31 as isize);
unsafe {
*__v96 = *__v96 ^ *subR.wrapping_offset(1 as isize);
}
// kw3
let __v99: *mut u32 = subL.wrapping_offset(32 as isize);
unsafe {
*__v99 = *__v99 ^ *subL.wrapping_offset(1 as isize);
}
let __v102: *mut u32 = subR.wrapping_offset(32 as isize);
unsafe {
*__v102 = *__v102 ^ *subR.wrapping_offset(1 as isize);
}
// absorb kw4 to other subkeys
kw4l = unsafe { *subL.wrapping_offset(33 as isize) };
kw4r = unsafe { *subR.wrapping_offset(33 as isize) };
// round 23
let __v105: *mut u32 = subL.wrapping_offset(30 as isize);
unsafe {
*__v105 = *__v105 ^ kw4l;
}
let __v108: *mut u32 = subR.wrapping_offset(30 as isize);
unsafe {
*__v108 = *__v108 ^ kw4r;
}
// round 21
let __v111: *mut u32 = subL.wrapping_offset(28 as isize);
unsafe {
*__v111 = *__v111 ^ kw4l;
}
let __v114: *mut u32 = subR.wrapping_offset(28 as isize);
unsafe {
*__v114 = *__v114 ^ kw4r;
}
// round 19
let __v117: *mut u32 = subL.wrapping_offset(26 as isize);
unsafe {
*__v117 = *__v117 ^ kw4l;
}
let __v120: *mut u32 = subR.wrapping_offset(26 as isize);
unsafe {
*__v120 = *__v120 ^ kw4r;
}
let __v124: u32 = kw4l ^ kw4r & !unsafe { *subR.wrapping_offset(24 as isize) };
kw4l = __v124;
dw = kw4l & unsafe { *subL.wrapping_offset(24 as isize) };
let __v125: u32 = rol32(dw, 1u32);
kw4r ^= __v125; // modified for FL(kl5)
}
// round 17
let __v128: *mut u32 = subL.wrapping_offset(22 as isize);
unsafe {
*__v128 = *__v128 ^ kw4l;
}
let __v131: *mut u32 = subR.wrapping_offset(22 as isize);
unsafe {
*__v131 = *__v131 ^ kw4r;
}
// round 15
let __v134: *mut u32 = subL.wrapping_offset(20 as isize);
unsafe {
*__v134 = *__v134 ^ kw4l;
}
let __v137: *mut u32 = subR.wrapping_offset(20 as isize);
unsafe {
*__v137 = *__v137 ^ kw4r;
}
// round 13
let __v140: *mut u32 = subL.wrapping_offset(18 as isize);
unsafe {
*__v140 = *__v140 ^ kw4l;
}
let __v143: *mut u32 = subR.wrapping_offset(18 as isize);
unsafe {
*__v143 = *__v143 ^ kw4r;
}
let __v147: u32 = kw4l ^ kw4r & !unsafe { *subR.wrapping_offset(16 as isize) };
kw4l = __v147;
dw = kw4l & unsafe { *subL.wrapping_offset(16 as isize) };
let __v148: u32 = rol32(dw, 1u32);
kw4r ^= __v148; // modified for FL(kl3)
// round 11
let __v151: *mut u32 = subL.wrapping_offset(14 as isize);
unsafe {
*__v151 = *__v151 ^ kw4l;
}
let __v154: *mut u32 = subR.wrapping_offset(14 as isize);
unsafe {
*__v154 = *__v154 ^ kw4r;
}
// round 9
let __v157: *mut u32 = subL.wrapping_offset(12 as isize);
unsafe {
*__v157 = *__v157 ^ kw4l;
}
let __v160: *mut u32 = subR.wrapping_offset(12 as isize);
unsafe {
*__v160 = *__v160 ^ kw4r;
}
// round 7
let __v163: *mut u32 = subL.wrapping_offset(10 as isize);
unsafe {
*__v163 = *__v163 ^ kw4l;
}
let __v166: *mut u32 = subR.wrapping_offset(10 as isize);
unsafe {
*__v166 = *__v166 ^ kw4r;
}
let __v170: u32 = kw4l ^ kw4r & !unsafe { *subR.wrapping_offset(8 as isize) };
kw4l = __v170;
dw = kw4l & unsafe { *subL.wrapping_offset(8 as isize) };
let __v171: u32 = rol32(dw, 1u32);
kw4r ^= __v171; // modified for FL(kl1)
// round 5
let __v174: *mut u32 = subL.wrapping_offset(6 as isize);
unsafe {
*__v174 = *__v174 ^ kw4l;
}
let __v177: *mut u32 = subR.wrapping_offset(6 as isize);
unsafe {
*__v177 = *__v177 ^ kw4r;
}
// round 3
let __v180: *mut u32 = subL.wrapping_offset(4 as isize);
unsafe {
*__v180 = *__v180 ^ kw4l;
}
let __v183: *mut u32 = subR.wrapping_offset(4 as isize);
unsafe {
*__v183 = *__v183 ^ kw4r;
}
// round 1
let __v186: *mut u32 = subL.wrapping_offset(2 as isize);
unsafe {
*__v186 = *__v186 ^ kw4l;
}
let __v189: *mut u32 = subR.wrapping_offset(2 as isize);
unsafe {
*__v189 = *__v189 ^ kw4r;
}
// kw1
let __v192: *mut u32 = subL.wrapping_offset(0 as isize);
unsafe {
*__v192 = *__v192 ^ kw4l;
}
let __v195: *mut u32 = subR.wrapping_offset(0 as isize);
unsafe {
*__v195 = *__v195 ^ kw4r;
}
// key XOR is end of F-function
unsafe {
*subkey.wrapping_offset(0i32 as isize) = *subL.wrapping_offset(0 as isize) ^ *subL.wrapping_offset(2 as isize);
}
// kw1
unsafe {
*subkey.wrapping_offset(1i32 as isize) = *subR.wrapping_offset(0 as isize) ^ *subR.wrapping_offset(2 as isize);
}
unsafe {
*subkey.wrapping_offset(4i32 as isize) = *subL.wrapping_offset(3 as isize);
}
// round 1
unsafe {
*subkey.wrapping_offset(5i32 as isize) = *subR.wrapping_offset(3 as isize);
}
unsafe {
*subkey.wrapping_offset(6i32 as isize) = *subL.wrapping_offset(2 as isize) ^ *subL.wrapping_offset(4 as isize);
}
// round 2
unsafe {
*subkey.wrapping_offset(7i32 as isize) = *subR.wrapping_offset(2 as isize) ^ *subR.wrapping_offset(4 as isize);
}
unsafe {
*subkey.wrapping_offset(8i32 as isize) = *subL.wrapping_offset(3 as isize) ^ *subL.wrapping_offset(5 as isize);
}
// round 3
unsafe {
*subkey.wrapping_offset(9i32 as isize) = *subR.wrapping_offset(3 as isize) ^ *subR.wrapping_offset(5 as isize);
}
unsafe {
*subkey.wrapping_offset(10i32 as isize) = *subL.wrapping_offset(4 as isize) ^ *subL.wrapping_offset(6 as isize);
}
// round 4
unsafe {
*subkey.wrapping_offset(11i32 as isize) = *subR.wrapping_offset(4 as isize) ^ *subR.wrapping_offset(6 as isize);
}
unsafe {
*subkey.wrapping_offset(12i32 as isize) = *subL.wrapping_offset(5 as isize) ^ *subL.wrapping_offset(7 as isize);
}
// round 5
unsafe {
*subkey.wrapping_offset(13i32 as isize) = *subR.wrapping_offset(5 as isize) ^ *subR.wrapping_offset(7 as isize);
}
tl = (unsafe { *subL.wrapping_offset(10 as isize) }) ^ (unsafe { *subR.wrapping_offset(10 as isize) }) & !unsafe { *subR.wrapping_offset(8 as isize) };
dw = tl & unsafe { *subL.wrapping_offset(8 as isize) }; // FL(kl1)
tr = (unsafe { *subR.wrapping_offset(10 as isize) }) ^ rol32(dw, 1u32);
unsafe {
*subkey.wrapping_offset(14i32 as isize) = *subL.wrapping_offset(6 as isize) ^ tl;
}
// round 6
unsafe {
*subkey.wrapping_offset(15i32 as isize) = *subR.wrapping_offset(6 as isize) ^ tr;
}
unsafe {
*subkey.wrapping_offset(16i32 as isize) = *subL.wrapping_offset(8 as isize);
}
// FL(kl1)
unsafe {
*subkey.wrapping_offset(17i32 as isize) = *subR.wrapping_offset(8 as isize);
}
unsafe {
*subkey.wrapping_offset(18i32 as isize) = *subL.wrapping_offset(9 as isize);
}
// FLinv(kl2)
unsafe {
*subkey.wrapping_offset(19i32 as isize) = *subR.wrapping_offset(9 as isize);
}
tl = (unsafe { *subL.wrapping_offset(7 as isize) }) ^ (unsafe { *subR.wrapping_offset(7 as isize) }) & !unsafe { *subR.wrapping_offset(9 as isize) };
dw = tl & unsafe { *subL.wrapping_offset(9 as isize) }; // FLinv(kl2)
tr = (unsafe { *subR.wrapping_offset(7 as isize) }) ^ rol32(dw, 1u32);
unsafe {
*subkey.wrapping_offset(20i32 as isize) = tl ^ *subL.wrapping_offset(11 as isize);
}
// round 7
unsafe {
*subkey.wrapping_offset(21i32 as isize) = tr ^ *subR.wrapping_offset(11 as isize);
}
unsafe {
*subkey.wrapping_offset(22i32 as isize) = *subL.wrapping_offset(10 as isize) ^ *subL.wrapping_offset(12 as isize);
}
// round 8
unsafe {
*subkey.wrapping_offset(23i32 as isize) = *subR.wrapping_offset(10 as isize) ^ *subR.wrapping_offset(12 as isize);
}
unsafe {
*subkey.wrapping_offset(24i32 as isize) = *subL.wrapping_offset(11 as isize) ^ *subL.wrapping_offset(13 as isize);
}
// round 9
unsafe {
*subkey.wrapping_offset(25i32 as isize) = *subR.wrapping_offset(11 as isize) ^ *subR.wrapping_offset(13 as isize);
}
unsafe {
*subkey.wrapping_offset(26i32 as isize) = *subL.wrapping_offset(12 as isize) ^ *subL.wrapping_offset(14 as isize);
}
// round 10
unsafe {
*subkey.wrapping_offset(27i32 as isize) = *subR.wrapping_offset(12 as isize) ^ *subR.wrapping_offset(14 as isize);
}
unsafe {
*subkey.wrapping_offset(28i32 as isize) = *subL.wrapping_offset(13 as isize) ^ *subL.wrapping_offset(15 as isize);
}
// round 11
unsafe {
*subkey.wrapping_offset(29i32 as isize) = *subR.wrapping_offset(13 as isize) ^ *subR.wrapping_offset(15 as isize);
}
tl = (unsafe { *subL.wrapping_offset(18 as isize) }) ^ (unsafe { *subR.wrapping_offset(18 as isize) }) & !unsafe { *subR.wrapping_offset(16 as isize) };
dw = tl & unsafe { *subL.wrapping_offset(16 as isize) }; // FL(kl3)
tr = (unsafe { *subR.wrapping_offset(18 as isize) }) ^ rol32(dw, 1u32);
unsafe {
*subkey.wrapping_offset(30i32 as isize) = *subL.wrapping_offset(14 as isize) ^ tl;
}
// round 12
unsafe {
*subkey.wrapping_offset(31i32 as isize) = *subR.wrapping_offset(14 as isize) ^ tr;
}
unsafe {
*subkey.wrapping_offset(32i32 as isize) = *subL.wrapping_offset(16 as isize);
}
// FL(kl3)
unsafe {
*subkey.wrapping_offset(33i32 as isize) = *subR.wrapping_offset(16 as isize);
}
unsafe {
*subkey.wrapping_offset(34i32 as isize) = *subL.wrapping_offset(17 as isize);
}
// FLinv(kl4)
unsafe {
*subkey.wrapping_offset(35i32 as isize) = *subR.wrapping_offset(17 as isize);
}
tl = (unsafe { *subL.wrapping_offset(15 as isize) }) ^ (unsafe { *subR.wrapping_offset(15 as isize) }) & !unsafe { *subR.wrapping_offset(17 as isize) };
dw = tl & unsafe { *subL.wrapping_offset(17 as isize) }; // FLinv(kl4)
tr = (unsafe { *subR.wrapping_offset(15 as isize) }) ^ rol32(dw, 1u32);
unsafe {
*subkey.wrapping_offset(36i32 as isize) = tl ^ *subL.wrapping_offset(19 as isize);
}
// round 13
unsafe {
*subkey.wrapping_offset(37i32 as isize) = tr ^ *subR.wrapping_offset(19 as isize);
}
unsafe {
*subkey.wrapping_offset(38i32 as isize) = *subL.wrapping_offset(18 as isize) ^ *subL.wrapping_offset(20 as isize);
}
// round 14
unsafe {
*subkey.wrapping_offset(39i32 as isize) = *subR.wrapping_offset(18 as isize) ^ *subR.wrapping_offset(20 as isize);
}
unsafe {
*subkey.wrapping_offset(40i32 as isize) = *subL.wrapping_offset(19 as isize) ^ *subL.wrapping_offset(21 as isize);
}
// round 15
unsafe {
*subkey.wrapping_offset(41i32 as isize) = *subR.wrapping_offset(19 as isize) ^ *subR.wrapping_offset(21 as isize);
}
unsafe {
*subkey.wrapping_offset(42i32 as isize) = *subL.wrapping_offset(20 as isize) ^ *subL.wrapping_offset(22 as isize);
}
// round 16
unsafe {
*subkey.wrapping_offset(43i32 as isize) = *subR.wrapping_offset(20 as isize) ^ *subR.wrapping_offset(22 as isize);
}
unsafe {
*subkey.wrapping_offset(44i32 as isize) = *subL.wrapping_offset(21 as isize) ^ *subL.wrapping_offset(23 as isize);
}
// round 17
unsafe {
*subkey.wrapping_offset(45i32 as isize) = *subR.wrapping_offset(21 as isize) ^ *subR.wrapping_offset(23 as isize);
}
if max == 24 {
unsafe {
*subkey.wrapping_offset(46i32 as isize) = *subL.wrapping_offset(22 as isize);
}
// round 18
unsafe {
*subkey.wrapping_offset(47i32 as isize) = *subR.wrapping_offset(22 as isize);
}
unsafe {
*subkey.wrapping_offset(48i32 as isize) = *subL.wrapping_offset(24 as isize) ^ *subL.wrapping_offset(23 as isize);
}
// kw3
unsafe {
*subkey.wrapping_offset(49i32 as isize) = *subR.wrapping_offset(24 as isize) ^ *subR.wrapping_offset(23 as isize);
}
} else {
tl = (unsafe { *subL.wrapping_offset(26 as isize) }) ^ (unsafe { *subR.wrapping_offset(26 as isize) }) & !unsafe { *subR.wrapping_offset(24 as isize) };
dw = tl & unsafe { *subL.wrapping_offset(24 as isize) }; // FL(kl5)
tr = (unsafe { *subR.wrapping_offset(26 as isize) }) ^ rol32(dw, 1u32);
unsafe {
*subkey.wrapping_offset(46i32 as isize) = *subL.wrapping_offset(22 as isize) ^ tl;
}
// round 18
unsafe {
*subkey.wrapping_offset(47i32 as isize) = *subR.wrapping_offset(22 as isize) ^ tr;
}
unsafe {
*subkey.wrapping_offset(48i32 as isize) = *subL.wrapping_offset(24 as isize);
}
// FL(kl5)
unsafe {
*subkey.wrapping_offset(49i32 as isize) = *subR.wrapping_offset(24 as isize);
}
unsafe {
*subkey.wrapping_offset(50i32 as isize) = *subL.wrapping_offset(25 as isize);
}
// FLinv(kl6)
unsafe {
*subkey.wrapping_offset(51i32 as isize) = *subR.wrapping_offset(25 as isize);
}
tl = (unsafe { *subL.wrapping_offset(23 as isize) }) ^ (unsafe { *subR.wrapping_offset(23 as isize) }) & !unsafe { *subR.wrapping_offset(25 as isize) };
dw = tl & unsafe { *subL.wrapping_offset(25 as isize) }; // FLinv(kl6)
tr = (unsafe { *subR.wrapping_offset(23 as isize) }) ^ rol32(dw, 1u32);
unsafe {
*subkey.wrapping_offset(52i32 as isize) = tl ^ *subL.wrapping_offset(27 as isize);
}
// round 19
unsafe {
*subkey.wrapping_offset(53i32 as isize) = tr ^ *subR.wrapping_offset(27 as isize);
}
unsafe {
*subkey.wrapping_offset(54i32 as isize) = *subL.wrapping_offset(26 as isize) ^ *subL.wrapping_offset(28 as isize);
}
// round 20
unsafe {
*subkey.wrapping_offset(55i32 as isize) = *subR.wrapping_offset(26 as isize) ^ *subR.wrapping_offset(28 as isize);
}
unsafe {
*subkey.wrapping_offset(56i32 as isize) = *subL.wrapping_offset(27 as isize) ^ *subL.wrapping_offset(29 as isize);
}
// round 21
unsafe {
*subkey.wrapping_offset(57i32 as isize) = *subR.wrapping_offset(27 as isize) ^ *subR.wrapping_offset(29 as isize);
}
unsafe {
*subkey.wrapping_offset(58i32 as isize) = *subL.wrapping_offset(28 as isize) ^ *subL.wrapping_offset(30 as isize);
}
// round 22
unsafe {
*subkey.wrapping_offset(59i32 as isize) = *subR.wrapping_offset(28 as isize) ^ *subR.wrapping_offset(30 as isize);
}
unsafe {
*subkey.wrapping_offset(60i32 as isize) = *subL.wrapping_offset(29 as isize) ^ *subL.wrapping_offset(31 as isize);
}
// round 23
unsafe {
*subkey.wrapping_offset(61i32 as isize) = *subR.wrapping_offset(29 as isize) ^ *subR.wrapping_offset(31 as isize);
}
unsafe {
*subkey.wrapping_offset(62i32 as isize) = *subL.wrapping_offset(30 as isize);
}
// round 24
unsafe {
*subkey.wrapping_offset(63i32 as isize) = *subR.wrapping_offset(30 as isize);
}
unsafe {
*subkey.wrapping_offset(64i32 as isize) = *subL.wrapping_offset(32 as isize) ^ *subL.wrapping_offset(31 as isize);
}
// kw3
unsafe {
*subkey.wrapping_offset(65i32 as isize) = *subR.wrapping_offset(32 as isize) ^ *subR.wrapping_offset(31 as isize);
}
}
}

fn camellia_setup128(key: *const u8, subkey: *mut u32) {
let mut kll: u32 = 0;
let mut klr: u32 = 0;
let mut krl: u32 = 0;
let mut krr: u32 = 0;
let mut il: u32 = 0;
let mut ir: u32 = 0;
let mut t0: u32 = 0;
let mut t1: u32 = 0;
let mut w0: u32 = 0;
let mut w1: u32 = 0;
let mut subL: __SlateAlign16<[u32; 26]> = __SlateAlign16([0u32; 26]);
let mut subR: __SlateAlign16<[u32; 26]> = __SlateAlign16([0u32; 26]);
// k == kll || klr || krl || krr (|| is concatenation)
kll = get_unaligned_be32(key as *const ());
klr = get_unaligned_be32(key.wrapping_offset(4 as isize) as *const ());
krl = get_unaligned_be32(key.wrapping_offset(8 as isize) as *const ());
krr = get_unaligned_be32(key.wrapping_offset(12 as isize) as *const ());
// generate KL dependent subkeys
//
// kw1
subL.0[0] = kll;
subR.0[0] = klr;
// kw2
subL.0[1] = krl;
subR.0[1] = krr;
// rotation left shift 15bit
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// k3
subL.0[4] = kll;
subR.0[4] = klr;
// k4
subL.0[5] = krl;
subR.0[5] = krr;
// rotation left shift 15+30bit
ROLDQ!(kll, klr, krl, krr, w0, 30i32);
// k7
subL.0[10] = kll;
subR.0[10] = klr;
// k8
subL.0[11] = krl;
subR.0[11] = krr;
// rotation left shift 15+30+15bit
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// k10
subL.0[13] = krl;
subR.0[13] = krr;
// rotation left shift 15+30+15+17 bit
ROLDQ!(kll, klr, krl, krr, w0, 17i32);
// kl3
subL.0[16] = kll;
subR.0[16] = klr;
// kl4
subL.0[17] = krl;
subR.0[17] = krr;
// rotation left shift 15+30+15+17+17 bit
ROLDQ!(kll, klr, krl, krr, w0, 17i32);
// k13
subL.0[18] = kll;
subR.0[18] = klr;
// k14
subL.0[19] = krl;
subR.0[19] = krr;
// rotation left shift 15+30+15+17+17+17 bit
ROLDQ!(kll, klr, krl, krr, w0, 17i32);
// k17
subL.0[22] = kll;
subR.0[22] = klr;
// k18
subL.0[23] = krl;
subR.0[23] = krr;
// generate KA
kll = subL.0[0];
klr = subR.0[0];
krl = subL.0[1];
krr = subR.0[1];
CAMELLIA_F!(kll, klr, 2694735487i64, 1003262091i64, w0, w1, il, ir, t0, t1);
krl ^= w0;
krr ^= w1;
CAMELLIA_F!(krl, krr, 3061508184i64, 1286239154i64, kll, klr, il, ir, t0, t1);
// current status == (kll, klr, w0, w1)
CAMELLIA_F!(kll, klr, 3337565999i64, 3914302142i64, krl, krr, il, ir, t0, t1);
krl ^= w0;
krr ^= w1;
CAMELLIA_F!(krl, krr, 1426019237i64, 4057165596i64, w0, w1, il, ir, t0, t1);
kll ^= w0;
klr ^= w1;
// generate KA dependent subkeys
//
// k1, k2
subL.0[2] = kll;
subR.0[2] = klr;
subL.0[3] = krl;
subR.0[3] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// k5,k6
subL.0[6] = kll;
subR.0[6] = klr;
subL.0[7] = krl;
subR.0[7] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// kl1, kl2
subL.0[8] = kll;
subR.0[8] = klr;
subL.0[9] = krl;
subR.0[9] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// k9
subL.0[12] = kll;
subR.0[12] = klr;
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// k11, k12
subL.0[14] = kll;
subR.0[14] = klr;
subL.0[15] = krl;
subR.0[15] = krr;
ROLDQo32!(kll, klr, krl, krr, w0, w1, 34i32);
// k15, k16
subL.0[20] = kll;
subR.0[20] = klr;
subL.0[21] = krl;
subR.0[21] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 17i32);
// kw3, kw4
subL.0[24] = kll;
subR.0[24] = klr;
subL.0[25] = krl;
subR.0[25] = krr;
camellia_setup_tail(subkey, subL.0.as_mut_ptr() as *mut u32, subR.0.as_mut_ptr() as *mut u32, 24i32);
}

fn camellia_setup256(key: *const u8, subkey: *mut u32) {
let mut kll: u32 = 0;
let mut klr: u32 = 0;
let mut krl: u32 = 0;
let mut krr: u32 = 0; // left half of key
let mut krll: u32 = 0;
let mut krlr: u32 = 0;
let mut krrl: u32 = 0;
let mut krrr: u32 = 0; // right half of key
let mut il: u32 = 0;
let mut ir: u32 = 0;
let mut t0: u32 = 0;
let mut t1: u32 = 0;
let mut w0: u32 = 0;
let mut w1: u32 = 0; // temporary variables
let mut subL: __SlateAlign16<[u32; 34]> = __SlateAlign16([0u32; 34]);
let mut subR: __SlateAlign16<[u32; 34]> = __SlateAlign16([0u32; 34]);
// key = (kll || klr || krl || krr || krll || krlr || krrl || krrr)
// (|| is concatenation)
kll = get_unaligned_be32(key as *const ());
klr = get_unaligned_be32(key.wrapping_offset(4 as isize) as *const ());
krl = get_unaligned_be32(key.wrapping_offset(8 as isize) as *const ());
krr = get_unaligned_be32(key.wrapping_offset(12 as isize) as *const ());
krll = get_unaligned_be32(key.wrapping_offset(16 as isize) as *const ());
krlr = get_unaligned_be32(key.wrapping_offset(20 as isize) as *const ());
krrl = get_unaligned_be32(key.wrapping_offset(24 as isize) as *const ());
krrr = get_unaligned_be32(key.wrapping_offset(28 as isize) as *const ());
// generate KL dependent subkeys
//
// kw1
subL.0[0] = kll;
subR.0[0] = klr;
// kw2
subL.0[1] = krl;
subR.0[1] = krr;
ROLDQo32!(kll, klr, krl, krr, w0, w1, 45i32);
// k9
subL.0[12] = kll;
subR.0[12] = klr;
// k10
subL.0[13] = krl;
subR.0[13] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// kl3
subL.0[16] = kll;
subR.0[16] = klr;
// kl4
subL.0[17] = krl;
subR.0[17] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 17i32);
// k17
subL.0[22] = kll;
subR.0[22] = klr;
// k18
subL.0[23] = krl;
subR.0[23] = krr;
ROLDQo32!(kll, klr, krl, krr, w0, w1, 34i32);
// k23
subL.0[30] = kll;
subR.0[30] = klr;
// k24
subL.0[31] = krl;
subR.0[31] = krr;
// generate KR dependent subkeys
ROLDQ!(krll, krlr, krrl, krrr, w0, 15i32);
// k3
subL.0[4] = krll;
subR.0[4] = krlr;
// k4
subL.0[5] = krrl;
subR.0[5] = krrr;
ROLDQ!(krll, krlr, krrl, krrr, w0, 15i32);
// kl1
subL.0[8] = krll;
subR.0[8] = krlr;
// kl2
subL.0[9] = krrl;
subR.0[9] = krrr;
ROLDQ!(krll, krlr, krrl, krrr, w0, 30i32);
// k13
subL.0[18] = krll;
subR.0[18] = krlr;
// k14
subL.0[19] = krrl;
subR.0[19] = krrr;
ROLDQo32!(krll, krlr, krrl, krrr, w0, w1, 34i32);
// k19
subL.0[26] = krll;
subR.0[26] = krlr;
// k20
subL.0[27] = krrl;
subR.0[27] = krrr;
ROLDQo32!(krll, krlr, krrl, krrr, w0, w1, 34i32);
// generate KA
kll = subL.0[0] ^ krll;
klr = subR.0[0] ^ krlr;
krl = subL.0[1] ^ krrl;
krr = subR.0[1] ^ krrr;
CAMELLIA_F!(kll, klr, 2694735487i64, 1003262091i64, w0, w1, il, ir, t0, t1);
krl ^= w0;
krr ^= w1;
CAMELLIA_F!(krl, krr, 3061508184i64, 1286239154i64, kll, klr, il, ir, t0, t1);
kll ^= krll;
klr ^= krlr;
CAMELLIA_F!(kll, klr, 3337565999i64, 3914302142i64, krl, krr, il, ir, t0, t1);
krl ^= w0 ^ krrl;
krr ^= w1 ^ krrr;
CAMELLIA_F!(krl, krr, 1426019237i64, 4057165596i64, w0, w1, il, ir, t0, t1);
kll ^= w0;
klr ^= w1;
// generate KB
krll ^= kll;
krlr ^= klr;
krrl ^= krl;
krrr ^= krr;
CAMELLIA_F!(krll, krlr, 283453434i64, 3731369245i64, w0, w1, il, ir, t0, t1);
krrl ^= w0;
krrr ^= w1;
CAMELLIA_F!(krrl, krrr, 2958461122i64, 3018244605i64, w0, w1, il, ir, t0, t1);
krll ^= w0;
krlr ^= w1;
// generate KA dependent subkeys
ROLDQ!(kll, klr, krl, krr, w0, 15i32);
// k5
subL.0[6] = kll;
subR.0[6] = klr;
// k6
subL.0[7] = krl;
subR.0[7] = krr;
ROLDQ!(kll, klr, krl, krr, w0, 30i32);
// k11
subL.0[14] = kll;
subR.0[14] = klr;
// k12
subL.0[15] = krl;
subR.0[15] = krr;
// rotation left shift 32bit
//
// kl5
subL.0[24] = klr;
subR.0[24] = krl;
// kl6
subL.0[25] = krr;
subR.0[25] = kll;
// rotation left shift 49 from k11,k12 -> k21,k22
ROLDQo32!(kll, klr, krl, krr, w0, w1, 49i32);
// k21
subL.0[28] = kll;
subR.0[28] = klr;
// k22
subL.0[29] = krl;
subR.0[29] = krr;
// generate KB dependent subkeys
//
// k1
subL.0[2] = krll;
subR.0[2] = krlr;
// k2
subL.0[3] = krrl;
subR.0[3] = krrr;
ROLDQ!(krll, krlr, krrl, krrr, w0, 30i32);
// k7
subL.0[10] = krll;
subR.0[10] = krlr;
// k8
subL.0[11] = krrl;
subR.0[11] = krrr;
ROLDQ!(krll, krlr, krrl, krrr, w0, 30i32);
// k15
subL.0[20] = krll;
subR.0[20] = krlr;
// k16
subL.0[21] = krrl;
subR.0[21] = krrr;
ROLDQo32!(krll, krlr, krrl, krrr, w0, w1, 51i32);
// kw3
subL.0[32] = krll;
subR.0[32] = krlr;
// kw4
subL.0[33] = krrl;
subR.0[33] = krrr;
camellia_setup_tail(subkey, subL.0.as_mut_ptr() as *mut u32, subR.0.as_mut_ptr() as *mut u32, 32i32);
}

fn camellia_setup192(key: *const u8, subkey: *mut u32) {
let mut kk: __SlateAlign16<[u8; 32]> = __SlateAlign16([0u8; 32]);
let mut krll: u32 = 0;
let mut krlr: u32 = 0;
let mut krrl: u32 = 0;
let mut krrr: u32 = 0;
unsafe { memcpy(kk.0.as_mut_ptr() as *mut (), key as *const (), 24u64) };
unsafe { memcpy((core::ptr::addr_of_mut!(krll) as *mut u8) as *mut (), key.wrapping_offset(16 as isize) as *const (), 4u64) };
unsafe { memcpy((core::ptr::addr_of_mut!(krlr) as *mut u8) as *mut (), key.wrapping_offset(20 as isize) as *const (), 4u64) };
krrl = !krll;
krrr = !krlr;
unsafe { memcpy((kk.0.as_mut_ptr() as *mut u8).wrapping_offset(24 as isize) as *mut (), (core::ptr::addr_of_mut!(krrl) as *mut u8) as *const (), 4u64) };
unsafe { memcpy((kk.0.as_mut_ptr() as *mut u8).wrapping_offset(28 as isize) as *mut (), (core::ptr::addr_of_mut!(krrr) as *mut u8) as *const (), 4u64) };
camellia_setup256(kk.0.as_mut_ptr() as *const u8, subkey);
}

// Encrypt/decrypt
/// max = 24: 128bit encrypt, max = 32: 256bit encrypt
fn camellia_do_encrypt(mut subkey: *const u32, mut io: *mut u32, max: u32) {
let mut il: u32 = 0;
let mut ir: u32 = 0;
let mut t0: u32 = 0;
let mut t1: u32 = 0; // temporary variables
// pre whitening but absorb kw2
let __v0: *mut u32 = io.wrapping_offset(0 as isize);
let __v1: u32 = unsafe { *__v0 };
let __v2: u32 = __v1 ^ SUBKEY_L!(0i32, subkey);
unsafe {
*__v0 = __v2;
}
let __v3: *mut u32 = io.wrapping_offset(1 as isize);
let __v4: u32 = unsafe { *__v3 };
let __v5: u32 = __v4 ^ SUBKEY_R!(0i32, subkey);
unsafe {
*__v3 = __v5;
}
// main iteration
ROUNDS!(0i32, io, subkey, ir, il);
FLS__2!(8i32, t0, subkey, il, io, ir, t1);
ROUNDS!(8i32, io, subkey, ir, il);
FLS__2!(16i32, t0, subkey, il, io, ir, t1);
ROUNDS!(16i32, io, subkey, ir, il);
if max == 32 {
FLS__2!(24i32, t0, subkey, il, io, ir, t1);
ROUNDS!(24i32, io, subkey, ir, il);
}
// post whitening but kw4
let __v754: *mut u32 = io.wrapping_offset(2 as isize);
let __v755: u32 = unsafe { *__v754 };
let __v756: u32 = __v755 ^ SUBKEY_L__2!(max, subkey);
unsafe {
*__v754 = __v756;
}
let __v757: *mut u32 = io.wrapping_offset(3 as isize);
let __v758: u32 = unsafe { *__v757 };
let __v759: u32 = __v758 ^ SUBKEY_R__2!(max, subkey);
unsafe {
*__v757 = __v759;
}
// NB: io[0],[1] should be swapped with [2],[3] by caller!
}

fn camellia_do_decrypt(mut subkey: *const u32, mut io: *mut u32, i: u32) {
let mut il: u32 = 0;
let mut ir: u32 = 0;
let mut t0: u32 = 0;
let mut t1: u32 = 0; // temporary variables
// pre whitening but absorb kw2
let __v0: *mut u32 = io.wrapping_offset(0 as isize);
let __v1: u32 = unsafe { *__v0 };
let __v2: u32 = __v1 ^ SUBKEY_L__2!(i, subkey);
unsafe {
*__v0 = __v2;
}
let __v3: *mut u32 = io.wrapping_offset(1 as isize);
let __v4: u32 = unsafe { *__v3 };
let __v5: u32 = __v4 ^ SUBKEY_R__2!(i, subkey);
unsafe {
*__v3 = __v5;
}
// main iteration
if i == 32 {
ROUNDS__2!(24i32, io, subkey, ir, il);
FLS!(24i32, t0, subkey, il, io, ir, t1);
}
ROUNDS__2!(16i32, io, subkey, ir, il);
FLS!(16i32, t0, subkey, il, io, ir, t1);
ROUNDS__2!(8i32, io, subkey, ir, il);
FLS!(8i32, t0, subkey, il, io, ir, t1);
ROUNDS__2!(0i32, io, subkey, ir, il);
// post whitening but kw4
let __v754: *mut u32 = io.wrapping_offset(2 as isize);
let __v755: u32 = unsafe { *__v754 };
let __v756: u32 = __v755 ^ SUBKEY_L!(0i32, subkey);
unsafe {
*__v754 = __v756;
}
let __v757: *mut u32 = io.wrapping_offset(3 as isize);
let __v758: u32 = unsafe { *__v757 };
let __v759: u32 = __v758 ^ SUBKEY_R!(0i32, subkey);
unsafe {
*__v757 = __v759;
}
// NB: 0,1 should be swapped with 2,3 by caller!
}

#[repr(C)]
#[derive(Clone, Copy)]
struct camellia_ctx {
    key_length: i32,
    key_table: [u32; 68],
}


#[unsafe(link_section = ".text.slate_distinct.camellia_generic.camellia_set_key")]
extern "C" fn camellia_set_key(tfm: *mut crypto_tfm, key: *const u8, key_len: u32) -> i32 {
let cctx: *mut camellia_ctx = crypto_tfm_ctx(tfm) as *mut camellia_ctx;
if key_len != 16 && key_len != 24 && key_len != 32 {
return 22i32.wrapping_neg();
}
unsafe {
(*cctx).key_length = key_len as i32;
}
match key_len {
16 => {
camellia_setup128(key, unsafe { (*cctx).key_table.as_mut_ptr() as *mut u32 });
}
24 => {
camellia_setup192(key, unsafe { (*cctx).key_table.as_mut_ptr() as *mut u32 });
}
32 => {
camellia_setup256(key, unsafe { (*cctx).key_table.as_mut_ptr() as *mut u32 });
}
_ => {
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.camellia_generic.camellia_encrypt")]
extern "C" fn camellia_encrypt(tfm: *mut crypto_tfm, out: *mut u8, r#in: *const u8) {
let cctx: *const camellia_ctx = crypto_tfm_ctx(tfm) as *const camellia_ctx;
let mut max: u32 = 0;
let mut tmp: __SlateAlign16<[u32; 4]> = __SlateAlign16([0u32; 4]);
tmp.0[0] = get_unaligned_be32(r#in as *const ());
tmp.0[1] = get_unaligned_be32(r#in.wrapping_offset(4 as isize) as *const ());
tmp.0[2] = get_unaligned_be32(r#in.wrapping_offset(8 as isize) as *const ());
tmp.0[3] = get_unaligned_be32(r#in.wrapping_offset(12 as isize) as *const ());
if (unsafe { (*cctx).key_length }) == 16 {
max = 24;
} else {
max = 32;
}
// for key lengths of 24 and 32
camellia_do_encrypt(unsafe { (*cctx).key_table.as_ptr() as *const u32 }, tmp.0.as_mut_ptr() as *mut u32, max);
// do_encrypt returns 0,1 swapped with 2,3
put_unaligned_be32(tmp.0[2], out as *mut ());
put_unaligned_be32(tmp.0[3], out.wrapping_offset(4 as isize) as *mut ());
put_unaligned_be32(tmp.0[0], out.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be32(tmp.0[1], out.wrapping_offset(12 as isize) as *mut ());
}

#[unsafe(link_section = ".text.slate_distinct.camellia_generic.camellia_decrypt")]
extern "C" fn camellia_decrypt(tfm: *mut crypto_tfm, out: *mut u8, r#in: *const u8) {
let cctx: *const camellia_ctx = crypto_tfm_ctx(tfm) as *const camellia_ctx;
let mut max: u32 = 0;
let mut tmp: __SlateAlign16<[u32; 4]> = __SlateAlign16([0u32; 4]);
tmp.0[0] = get_unaligned_be32(r#in as *const ());
tmp.0[1] = get_unaligned_be32(r#in.wrapping_offset(4 as isize) as *const ());
tmp.0[2] = get_unaligned_be32(r#in.wrapping_offset(8 as isize) as *const ());
tmp.0[3] = get_unaligned_be32(r#in.wrapping_offset(12 as isize) as *const ());
if (unsafe { (*cctx).key_length }) == 16 {
max = 24;
} else {
max = 32;
}
// for key lengths of 24 and 32
camellia_do_decrypt(unsafe { (*cctx).key_table.as_ptr() as *const u32 }, tmp.0.as_mut_ptr() as *mut u32, max);
// do_decrypt returns 0,1 swapped with 2,3
put_unaligned_be32(tmp.0[2], out as *mut ());
put_unaligned_be32(tmp.0[3], out.wrapping_offset(4 as isize) as *mut ());
put_unaligned_be32(tmp.0[0], out.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be32(tmp.0[1], out.wrapping_offset(12 as isize) as *mut ());
}

static mut camellia_alg: crypto_alg = {
let mut __t0: crypto_alg = unsafe { core::mem::zeroed() };
__t0.cra_flags = 1u32;
__t0.cra_blocksize = 16u32;
__t0.cra_ctxsize = (core::mem::size_of::<camellia_ctx>() as u64) as u32;
__t0.cra_priority = 100i32;
__t0.cra_name = [99u8, 97u8, 109u8, 101u8, 108u8, 108u8, 105u8, 97u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.cra_driver_name = [99u8, 97u8, 109u8, 101u8, 108u8, 108u8, 105u8, 97u8, 45u8, 103u8, 101u8, 110u8, 101u8, 114u8, 105u8, 99u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t0.cra_u = {
let mut __t1: __SlateRecord_crypto_h_350 = unsafe { core::mem::zeroed() };
__t1.cipher = cipher_alg { cia_min_keysize: 16u32, cia_max_keysize: 32u32, cia_setkey: Some(camellia_set_key), cia_encrypt: Some(camellia_encrypt), cia_decrypt: Some(camellia_decrypt) };
__t1
};
__t0.cra_module = core::ptr::null_mut::<module>();
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn camellia_init() -> i32 {
unsafe { crypto_register_alg(core::ptr::addr_of_mut!(camellia_alg)) }
}

#[unsafe(link_section = ".exit.text")]
extern "C" fn camellia_fini() {
unsafe { crypto_unregister_alg(core::ptr::addr_of_mut!(camellia_alg)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_camellia_init_442: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(camellia_init)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_camellia_fini: Option<unsafe extern "C" fn()> = Some(camellia_fini);

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_443: [u8; 55] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 67, 97, 109, 101, 108, 108, 105, 97, 32, 67, 105, 112, 104, 101, 114, 32, 65, 108, 103, 111, 114, 105, 116, 104, 109, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_444: [u8; 46] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 102, 105, 108, 101, 61, 99, 114, 121, 112, 116, 111, 47, 99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_445: [u8; 29] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_446: [u8; 32] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 97, 108, 105, 97, 115, 61, 99, 97, 109, 101, 108, 108, 105, 97, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_447: [u8; 39] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 97, 108, 105, 97, 115, 61, 99, 114, 121, 112, 116, 111, 45, 99, 97, 109, 101, 108, 108, 105, 97, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_448: [u8; 40] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 97, 108, 105, 97, 115, 61, 99, 97, 109, 101, 108, 108, 105, 97, 45, 103, 101, 110, 101, 114, 105, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_449: [u8; 47] = [99, 97, 109, 101, 108, 108, 105, 97, 95, 103, 101, 110, 101, 114, 105, 99, 46, 97, 108, 105, 97, 115, 61, 99, 114, 121, 112, 116, 111, 45, 99, 97, 109, 101, 108, 108, 105, 97, 45, 103, 101, 110, 101, 114, 105, 99, 0];
