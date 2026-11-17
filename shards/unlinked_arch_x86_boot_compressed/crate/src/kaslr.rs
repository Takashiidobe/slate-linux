
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
//! SPDX-License-Identifier: GPL-2.0
//!
//! When building position independent code with GCC using the -fPIC option,
//! (or even the -fPIE one on older versions), it will assume that we are
//! building a dynamic object (either a shared library or an executable) that
//! may have symbol references that can only be resolved at load time. For a
//! variety of reasons (ELF symbol preemption, the CoW footprint of the section
//! that is modified by the loader), this results in all references to symbols
//! with external linkage to go via entries in the Global Offset Table (GOT),
//! which carries absolute addresses which need to be fixed up when the
//! executable image is loaded at an offset which is different from its link
//! time offset.
//!
//! Fortunately, there is a way to inform the compiler that such symbol
//! references will be satisfied at link time rather than at load time, by
//! giving them 'hidden' visibility.
//! SPDX-License-Identifier: GPL-2.0
//!
//! kaslr.c
//!
//! This contains the routines needed to generate a reasonable level of
//! entropy to choose a randomized kernel base address offset in support
//! of Kernel Address Space Layout Randomization (KASLR). Additionally
//! handles walking the physical memory maps (and tracking memory regions
//! to avoid) in order to select a physical memory location that can
//! contain the entire properly aligned running kernel image.
//! Boot stub deals with identity mappings, physical and virtual addresses are
//! the same, so override these defines.
//!
//! <asm/page.h> will not define them if they are already defined.
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
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! Allow symbol exports to be disabled completely so that C code may
//! be reused in other execution contexts such as the UEFI stub or the
//! decompressor.
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
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! Bitfields in the atomic value:
//!
//! When NR_CPUS < 16K
//!  0- 7: locked byte
//!  8-15: pending byte
//! 16-17: tail index
//! 18-31: tail cpu (+1)
//!
//! When NR_CPUS >= 16K
//!  0- 7: locked byte
//!     8: pending
//!  9-10: tail index
//! 11-31: tail cpu (+1)
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
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
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! label
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
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Physical address where kernel should be loaded.
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
//! Boot IDT definitions
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
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
//! 64bit FS base
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! signal pending
//! 32-bit address space on 64 bits
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
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
//! Physical address zone modifiers (see linux/mmzone.h - low four bits)
//!
//! Do not put any conditional on these. If necessary modify the definitions
//! without the underscores and use them consistently. The definitions here may
//! be used in bit comparisons.
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
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! If !LOCKDEP, the value is meaningless
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
//! Convenience macros for the sake of set_current_state:
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! Dependency order vs. p above.
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
//! wait_queue_entry::flags
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
//! Subtract two ktime_t variables. rem = lhs -rhs:
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! __MAX_NR_ZONES
//! percpu_counter batch for local add or sub
//! Opaque type representing current VMA (vm_area_struct) flag state. Must be
//! accessed via vma_flags_xxx() helper functions.
//! Desired design of maximum size and alignment (see RFC2553)
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
//! We encode errnos in the xas->xa_node.  If an error has happened, we need to
//! drop the lock to fix it, and once we've done so the xa_state is invalid.
//! The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
//! not including trailing NUL.
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
//! current_cred - Access the current task's subjective credentials
//!
//! Access the subjective credentials of the current task.  RCU-safe,
//! since nobody else can modify it.
//! group exit in progress
//! sb->s_flags.  Note that these mirror the equivalent MS_* flags where
//! represented in both.
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
//! 192 bytes
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! vfs{g,u}ids are created from k{g,u}ids.
//! We don't allow them to be created from regular {u,g}id.
//! no-op
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
//! number of edd_info structs starting at EDDBUF
//! Resource type
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Only touch one cacheline for common rbtree-for-core-layout case.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! upper
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! The maximum expected number of CSI-2 data lanes.
//!
//! This number is not expected to ever have to be equal to or greater than the
//! number of bits in an unsigned long variable, but if it needs to be increased
//! above that limit, code will need to be adjusted accordingly.
//! See the comment in asm-generic/io.h about ioremap_np().
//! DWORD 3
//! higher reliability
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
// generic data direction definitions
// The type of an index into the pagecache.
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
// reg := -EIO
// reg := -EIO
// modpost check for section mismatches during the kernel build.
// A section mismatch happens when there are references from a
// code or data section to an init section (both code or data).
// The init sections are (for most archs) discarded by the kernel
// when early init has completed so all such references are potential bugs.
// For exit sections the same issue exists.
//
// The following markers are used for the cases where the reference to
// the *init / *exit section (code or data) is valid and will teach
// modpost not to issue a warning.  Intended semantics is that a code or
// data tagged __ref* can reference code or data from init section without
// producing a warning (of course, no warning does not mean code is
// correct, so optimally document why the __ref is needed and why it's OK).
//
// The markers follow same syntax rules as __init / __initdata.
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
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
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// Operation not supported on transport endpoint
// Operation is not supported
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
// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
// @ptr and is invoked once before a percpu area is accessed by all
// accessors and operations.  This is performed in the generic part of
// percpu and arch overrides don't need to worry about it; however, if an
// arch wants to implement an arch-specific percpu accessor or operation,
// it may use __verify_pcpu_ptr() to verify the parameters.
//
// + 0 is required in order to convert the pointer type from a
// potential array type to a pointer to a single item of the array.
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
// "popcnt" POPCNT instruction
// "rdrand" RDRAND instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Xen paravirtual guest
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "rdseed" RDSEED instruction
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "hwp" Intel Hardware P-states
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
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
// No need to mask any bits for !PAE
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
// N 32-bit words worth of info
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
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
// Same as rb_first(), but O(1)
// 256 bytes including ->parent
// 256 bytes
// 240 bytes
// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
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
// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
// Implementation specific max size
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
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
// Mount read-only
// Update the on-disk [acm]times lazily
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
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
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
// non-RWF related bits - start at 16
// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
// File system supports uncached read/write buffered IO
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
// Resource extended types
// No address assigned yet
// Set the IDR flag and the IDR_FREE tag
// buffer for the variables
// Generate SD flag bits
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// lower
// digit
// cntrl
// punct
// white space (space/lf/tab)
// hex digit
// hard space (0x20)
// Fixed by ACPI spec
// DWORD 5
// Various well-known legacy memory ranges in physical memory:
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
macro_rules! ALIGN_DOWN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x.wrapping_sub($a.wrapping_sub(1)), $a)
};
}

/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

macro_rules! EAX_EDX_VAL {
($low:expr, $high:expr) => {
($low) | ($high) << 32i32
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

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
($x) + $mask & !$mask
};
}

macro_rules! __ALIGN_KERNEL_MASK__2 {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK__2!($x, (($a) as u64).wrapping_sub(1))
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

macro_rules! __cmp_once__2 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
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

macro_rules! __ismask {
($x:expr) => {
unsafe { _ctype.0[((($x) as u8) as i32) as usize] }
};
}

macro_rules! __ismask__2 {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype.0) as *const u8).offset(((($x) as u8) as i32) as isize) }
};
}

macro_rules! __or {
() => {
____or!()
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
if is_signed_type!() { 2 + (__is_nonneg!($ux) as i32) } else { 1 + 2 * ((((const { core::mem::size_of::<u64>() }) as u64) < 4) as i32) }
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

/// check_shl_overflow() - Calculate a left-shifted value and check overflow
/// @a: Value to be shifted
/// @s: How many bits left to shift
/// @d: Pointer to where to store the result
///
/// Computes *@d = (@a << @s)
///
/// Returns true if '*@d' cannot hold the result or when '@a << @s' doesn't
/// make sense. Example conditions:
///
/// - '@a << @s' causes bits to be lost when stored in *@d.
/// - '@s' is garbage (e.g. negative) or so large that the result of
///   '@a << @s' is guaranteed to be 0.
/// - '@a' is negative.
/// - '@a << @s' sets the sign bit, if any, in '*@d'.
///
/// '*@d' will hold the results of the attempted shift, but is not
/// considered "safe for use" if true is returned.
macro_rules! check_shl_overflow {
($a:expr, $s:expr, $d:expr) => {
{
let mut __v: bool;
let __v2: u64 = $a;
let __v3: u32 = $s;
let __v4: *mut u64 = $d;
let __v5: u64 = __v2;
let __v6: u32 = if __v3 >= 0 && (__v3 as u64) < 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64) { __v3 } else { 0 };
unsafe {
*__v4 = __v5 << __v6;
}
__v = __v6 != __v3 || (unsafe { *__v4 }) < 0 || __v2 < 0 || (unsafe { *__v4 }) >> __v6 != __v2;
__must_check_overflow(__v)
}
};
}

macro_rules! debug_putstr__3 {
($__x:expr) => {
unsafe { __putstr($__x) };
};
}

macro_rules! debug_putstr {
($__x:expr) => {
unsafe { __putstr(($__x) as *const i8) };
};
}

macro_rules! efi_early_memdesc_ptr {
($map:expr, $desc_size:expr, $n:expr) => {
((unsafe { ((($map) as *mut ()) as *const u8).offset((($n) as u32).wrapping_mul($desc_size) as isize) }) as *mut ()) as *mut __SlateRecord_efi_h_51
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
18446744073709551615u64 < 1
};
}

/// Note: isspace() must return false for %NUL-terminator
macro_rules! isspace {
($c:expr) => {
(__ismask__2!($c) as i32) & 32 != 0
};
}

/// Note: isspace() must return false for %NUL-terminator
macro_rules! isspace__2 {
($c:expr) => {
(__ismask!($c) as i32) & 32 != 0
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

/// Access builtin version by default.
macro_rules! memcpy {
($d:expr, $s:expr, $l:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), $l) };
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

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut __pgtable_l5_enabled: u32;
    static mut pgdir_shift: u32;
    static mut ptrs_per_p4d: u32;
    static mut boot_params_ptr: *mut boot_params;
    static mut ist_info: ist_info;
    static mut pio_ops: port_io_ops;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn strlen(arg0: *const i8) -> u64;
    fn strcmp(arg0: *const i8, arg1: *const i8) -> i32;
    fn strncmp(arg0: *const i8, arg1: *const i8, arg2: u64) -> i32;
    fn strchr(arg0: *const i8, arg1: i32) -> *mut i8;
    fn strstr(arg0: *const i8, arg1: *const i8) -> *mut i8;
    fn strnlen(arg0: *const i8, arg1: u64) -> u64;
    fn simple_strtol(arg0: *const i8, arg1: *mut *mut i8, arg2: u32) -> i64;
    fn simple_strtoull(arg0: *const i8, arg1: *mut *mut i8, arg2: u32) -> u64;
    fn __efi_soft_reserve_enabled() -> bool;
    fn malloc(arg0: i32) -> *mut ();
    fn free(arg0: *mut ());
    fn __putstr(arg0: *const i8);
    fn cmdline_find_option_bool(arg0: *const i8) -> i32;
    fn has_cpuflag(arg0: i32) -> bool;
    fn warn(arg0: *const i8);
    fn error(arg0: *mut i8);
    fn boot_kstrtoul(arg0: *const i8, arg1: u32, arg2: *mut u64) -> i32;
    fn get_cmd_line_ptr() -> u64;
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn __compiletime_assert_465();
    fn __compiletime_assert_468();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct screen_info {
    orig_x: u8,
    orig_y: u8,
    ext_mem_k: u16,
    orig_video_page: u16,
    orig_video_mode: u8,
    orig_video_cols: u8,
    flags: u8,
    unused2: u8,
    orig_video_ega_bx: u16,
    unused3: u16,
    orig_video_lines: u8,
    orig_video_isVGA: u8,
    orig_video_points: u16,
    lfb_width: u16,
    lfb_height: u16,
    lfb_depth: u16,
    lfb_base: u32,
    lfb_size: u32,
    cl_magic: u16,
    cl_offset: u16,
    lfb_linelength: u16,
    red_size: u8,
    red_pos: u8,
    green_size: u8,
    green_pos: u8,
    blue_size: u8,
    blue_pos: u8,
    rsvd_size: u8,
    rsvd_pos: u8,
    vesapm_seg: u16,
    vesapm_off: u16,
    pages: u16,
    vesa_attributes: u16,
    capabilities: u32,
    ext_lfb_base: u32,
    _reserved: [u8; 2],
}


// Monochrome Text Display
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
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

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
fn pgtable_l5_enabled() -> bool {
(unsafe { __pgtable_l5_enabled }) != 0
}

// 4th level page in 5-level paging case
// 3rd level page
// entries per page directory level
// See Documentation/arch/x86/x86_64/mm.rst for a description of the memory map.
//
// Be very careful vs. KASLR when changing anything here. The KASLR address
// range must not overlap with anything except the KASAN shadow area, which
// is correct as KASAN disables KASLR.
// We borrow bit 3 to remember PG_anon_exclusive.
#[inline(always)]
fn rdtsc() -> u64 {
let mut low: u64 = 0;
let mut high: u64 = 0;
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("rdtsc", lateout("ax") __t0, lateout("dx") __t1, options(nostack, att_syntax));
low = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
high = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
EAX_EDX_VAL!(low, high)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct setup_data {
    next: u64,
    r#type: u32,
    len: u32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct setup_indirect {
    r#type: u32,
    reserved: u32,
    len: u64,
    addr: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct boot_e820_entry {
    addr: u64,
    size: u64,
    r#type: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct kho_data {
    fdt_addr: u64,
    fdt_size: u64,
    scratch_addr: u64,
    scratch_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_edd_h_87 {
    isa: __SlateRecord_edd_h_88,
    pci: __SlateRecord_edd_h_93,
    ibnd: __SlateRecord_edd_h_101,
    xprs: __SlateRecord_edd_h_104,
    htpt: __SlateRecord_edd_h_107,
    unknown: __SlateRecord_edd_h_110,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_88 {
    base_address: u16,
    reserved1: u16,
    reserved2: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_93 {
    bus: u8,
    slot: u8,
    function: u8,
    channel: u8,
    reserved: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_101 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_104 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_107 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_110 {
    reserved: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_edd_h_114 {
    ata: __SlateRecord_edd_h_115,
    atapi: __SlateRecord_edd_h_122,
    scsi: __SlateRecord_edd_h_130,
    usb: __SlateRecord_edd_h_136,
    i1394: __SlateRecord_edd_h_140,
    fibre: __SlateRecord_edd_h_144,
    i2o: __SlateRecord_edd_h_148,
    raid: __SlateRecord_edd_h_152,
    sata: __SlateRecord_edd_h_157,
    unknown: __SlateRecord_edd_h_164,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_115 {
    device: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_122 {
    device: u8,
    lun: u8,
    reserved1: u8,
    reserved2: u8,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_130 {
    id: u16,
    lun: u64,
    reserved1: u16,
    reserved2: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_136 {
    serial_number: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_140 {
    eui: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_144 {
    wwid: u64,
    lun: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_148 {
    identity_tag: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_152 {
    array_number: u32,
    reserved1: u32,
    reserved2: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_157 {
    device: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_164 {
    reserved1: u64,
    reserved2: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct apm_bios_info {
    version: u16,
    cseg: u16,
    offset: u32,
    cseg_16: u16,
    dseg: u16,
    flags: u16,
    cseg_len: u16,
    cseg_16_len: u16,
    dseg_len: u16,
}


// max number of signatures to store
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct edd_device_params {
    length: u16,
    info_flags: u16,
    num_default_cylinders: u32,
    num_default_heads: u32,
    sectors_per_track: u32,
    number_of_sectors: u64,
    bytes_per_sector: u16,
    dpte_ptr: u32,
    key: u16,
    device_path_info_length: u8,
    reserved2: u8,
    reserved3: u16,
    host_bus_type: [u8; 4],
    interface_type: [u8; 8],
    interface_path: __SlateRecord_edd_h_87,
    device_path: __SlateRecord_edd_h_114,
    reserved4: u8,
    checksum: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct edd_info {
    device: u8,
    version: u8,
    interface_support: u16,
    legacy_max_cylinder: u16,
    legacy_max_head: u8,
    legacy_sectors_per_track: u8,
    params: edd_device_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ist_info {
    signature: u32,
    command: u32,
    event: u32,
    perf_level: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct edid_info {
    dummy: [u8; 128],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct setup_header {
    setup_sects: u8,
    root_flags: u16,
    syssize: u32,
    ram_size: u16,
    vid_mode: u16,
    root_dev: u16,
    boot_flag: u16,
    jump: u16,
    header: u32,
    version: u16,
    realmode_swtch: u32,
    start_sys_seg: u16,
    kernel_version: u16,
    type_of_loader: u8,
    loadflags: u8,
    setup_move_size: u16,
    code32_start: u32,
    ramdisk_image: u32,
    ramdisk_size: u32,
    bootsect_kludge: u32,
    heap_end_ptr: u16,
    ext_loader_ver: u8,
    ext_loader_type: u8,
    cmd_line_ptr: u32,
    initrd_addr_max: u32,
    kernel_alignment: u32,
    relocatable_kernel: u8,
    min_alignment: u8,
    xloadflags: u16,
    cmdline_size: u32,
    hardware_subarch: u32,
    hardware_subarch_data: u64,
    payload_offset: u32,
    payload_length: u32,
    setup_data: u64,
    pref_address: u64,
    init_size: u32,
    handover_offset: u32,
    kernel_info_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sys_desc_table {
    length: u16,
    table: [u8; 14],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct olpc_ofw_header {
    ofw_magic: u32,
    ofw_version: u32,
    cif_handler: u32,
    irq_desc_table: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct efi_info {
    efi_loader_signature: u32,
    efi_systab: u32,
    efi_memdesc_size: u32,
    efi_memdesc_version: u32,
    efi_memmap: u32,
    efi_memmap_size: u32,
    efi_systab_hi: u32,
    efi_memmap_hi: u32,
}


// This is the maximum number of entries in struct boot_params::e820_table
// (the zeropage), which is part of the x86 boot protocol ABI:
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct boot_params {
    screen_info: screen_info,
    apm_bios_info: apm_bios_info,
    _pad2: [u8; 4],
    tboot_addr: u64,
    ist_info: ist_info,
    acpi_rsdp_addr: u64,
    _pad3: [u8; 8],
    hd0_info: [u8; 16],
    hd1_info: [u8; 16],
    sys_desc_table: sys_desc_table,
    olpc_ofw_header: olpc_ofw_header,
    ext_ramdisk_image: u32,
    ext_ramdisk_size: u32,
    ext_cmd_line_ptr: u32,
    _pad4: [u8; 112],
    cc_blob_address: u32,
    edid_info: edid_info,
    efi_info: efi_info,
    alt_mem_k: u32,
    scratch: u32,
    e820_entries: u8,
    eddbuf_entries: u8,
    edd_mbr_sig_buf_entries: u8,
    kbd_status: u8,
    secure_boot: u8,
    _pad5: [u8; 2],
    sentinel: u8,
    _pad6: [u8; 1],
    hdr: setup_header,
    _pad7: [u8; 36],
    edd_mbr_sig_buffer: [u32; 16],
    e820_table: [boot_e820_entry; 128],
    _pad8: [u8; 48],
    eddbuf: [edd_info; 6],
    _pad9: [u8; 276],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct port_io_ops {
    f_inb: Option<unsafe extern "C" fn(u16) -> u8>,
    f_outb: Option<unsafe extern "C" fn(u8, u16)>,
    f_outw: Option<unsafe extern "C" fn(u16, u16)>,
}


// Redirect port I/O operations via pio_ops callbacks.
// TDX guests override these callbacks with TDX-specific helpers.
// soft reserved
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_efi_h_51 {
    r#type: u32,
    pad: u32,
    phys_addr: u64,
    virt_addr: u64,
    num_pages: u64,
    attribute: u64,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

fn efi_soft_reserve_enabled() -> bool {
IS_ENABLED!() != 0 && unsafe { __efi_soft_reserve_enabled() }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_vector {
    start: u64,
    size: u64,
}


#[unsafe(no_mangle)]
extern "C" fn choose_random_location(input: u64, input_size: u64, output: *mut u64, output_size: u64, virt_addr: *mut u64) {
let mut random_addr: u64 = 0;
let mut min_addr: u64 = 0;
if (unsafe { cmdline_find_option_bool(c"nokaslr".as_ptr()) }) != 0 {
unsafe { warn(c"KASLR disabled: 'nokaslr' on cmdline.".as_ptr()) };
return;
}
let __v0: *mut boot_params = unsafe { boot_params_ptr };
unsafe {
(*__v0).hdr.loadflags = (((*__v0).hdr.loadflags as i32) | 2i32) as u8;
}
if IS_ENABLED!() != 0 {
unsafe {
mem_limit = 1073741824u64;
}
} else {
unsafe {
mem_limit = 1u64 << if pgtable_l5_enabled() { 52i32 } else { 46i32 };
}
}
// Record the various known unsafe memory ranges.
mem_avoid_init(input, input_size, unsafe { *output });
// Low end of the randomization range should be the
// smaller of 512M or the initial kernel image
// location:
let mut __v3: u64;
let __UNIQUE_ID_x__466: u64 = unsafe { *output };
let __UNIQUE_ID_y__467: u64 = 536870912u64;
loop {
if __types_ok!(__UNIQUE_ID_x__466, __UNIQUE_ID_y__467) == 0 {
unsafe { __compiletime_assert_468() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__466 < __UNIQUE_ID_y__467 { __UNIQUE_ID_x__466 } else { __UNIQUE_ID_y__467 };
min_addr = __v3;
// Make sure minimum is aligned.
min_addr = ALIGN!(min_addr, 2097152i32);
// Walk available memory entries to find a random address.
random_addr = find_random_phys_addr(min_addr, output_size);
if random_addr == 0 {
unsafe { warn(c"Physical KASLR disabled: no suitable memory region!".as_ptr()) };
} else {
// Update the new physical address location.
if (unsafe { *output }) != random_addr {
unsafe {
*output = random_addr;
}
}
}
// Pick random virtual address starting from LOAD_PHYSICAL_ADDR.
if IS_ENABLED__2!() != 0 {
random_addr = find_random_virt_addr(__ALIGN_KERNEL_MASK!(16777216i32, 2097151i32) as u64, output_size);
}
unsafe {
*virt_addr = random_addr;
}
}

fn count_immovable_mem_regions() -> i32 {
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kho_scratch {
    addr: u64,
    size: u64,
}


// isspace() in linux/ctype.h is expected by next_args() to filter
// out "space/lf/tab". While boot/ctype.h conflicts with linux/ctype.h,
// since isdigit() is implemented in both of them. Hence disable it
// here.
// For COMMAND_LINE_SIZE
/// Simplified build-specific string for starting entropy.
static build_str: __SlateAlign16<[i8; 143]> = __SlateAlign16([55i8, 46i8, 51i8, 46i8, 48i8, 45i8, 114i8, 99i8, 53i8, 45i8, 103i8, 54i8, 97i8, 100i8, 100i8, 98i8, 52i8, 102i8, 51i8, 56i8, 53i8, 53i8, 55i8, 32i8, 40i8, 116i8, 97i8, 107i8, 97i8, 115i8, 104i8, 105i8, 64i8, 99i8, 97i8, 99i8, 104i8, 121i8, 41i8, 32i8, 40i8, 99i8, 108i8, 97i8, 110i8, 103i8, 32i8, 118i8, 101i8, 114i8, 115i8, 105i8, 111i8, 110i8, 32i8, 50i8, 51i8, 46i8, 49i8, 46i8, 52i8, 44i8, 32i8, 71i8, 78i8, 85i8, 32i8, 108i8, 100i8, 32i8, 40i8, 71i8, 78i8, 85i8, 32i8, 66i8, 105i8, 110i8, 117i8, 116i8, 105i8, 108i8, 115i8, 41i8, 32i8, 50i8, 46i8, 52i8, 55i8, 41i8, 32i8, 35i8, 50i8, 32i8, 83i8, 77i8, 80i8, 32i8, 80i8, 82i8, 69i8, 69i8, 77i8, 80i8, 84i8, 95i8, 68i8, 89i8, 78i8, 65i8, 77i8, 73i8, 67i8, 32i8, 83i8, 117i8, 110i8, 32i8, 78i8, 111i8, 118i8, 32i8, 49i8, 53i8, 32i8, 48i8, 48i8, 58i8, 50i8, 48i8, 58i8, 53i8, 51i8, 32i8, 69i8, 83i8, 84i8, 32i8, 50i8, 48i8, 50i8, 54i8, 0i8]);

fn rotate_xor(mut hash: u64, area: *const (), size: u64) -> u64 {
let mut i__0: u64 = 0;
let ptr: *mut u64 = area as *mut u64;
while i__0 < size / (core::mem::size_of::<u64>() as u64) {
// Rotate by odd number of bits and XOR.
hash = hash << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(7) | hash >> 7i32;
hash ^= unsafe { *ptr.add(i__0 as usize) };
i__0 = i__0.wrapping_add(1);
}
hash
}

/// Attempt to create a simple but unpredictable starting entropy.
fn get_boot_seed() -> u64 {
let mut hash: u64 = 0;
hash = rotate_xor(hash, (unsafe { core::ptr::addr_of!(build_str.0) as *const i8 }) as *const (), core::mem::size_of::<[i8; 143]>() as u64);
hash = rotate_xor(hash, (unsafe { boot_params_ptr }) as *const (), core::mem::size_of::<boot_params>() as u64);
hash
}

fn rdrand_long(v: &mut u64) -> bool {
let mut ok: bool = false;
let mut retry: u32 = 10;
loop {
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("rdrand {1:r}\nsetc {0}", lateout(reg_byte) __t0, lateout(reg) __t1, options(nostack, att_syntax));
ok = (__t0 as u8) != 0;
*(v as *mut u64) = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
if ok {
return true;
}
let __v2: u32 = retry.wrapping_sub(1);
retry = __v2;
if __v2 == 0 {
break;
}
}
false
}

fn i8254() -> u16 {
let mut status: u16 = 0;
let mut timer: u16 = 0;
loop {
unsafe { unsafe { pio_ops.f_outb }.unwrap()(194u8, 67u16) };
status = (unsafe { unsafe { pio_ops.f_inb }.unwrap()(64u16) }) as u16;
timer = (unsafe { unsafe { pio_ops.f_inb }.unwrap()(64u16) }) as u16;
let __v0: i32 = ((unsafe { unsafe { pio_ops.f_inb }.unwrap()(64u16) }) as i32) << 8i32;
timer = ((timer as i32) | __v0) as u16;
if (status as i32) & 64 == 0 {
break;
}
}
timer
}

#[unsafe(no_mangle)]
extern "C" fn kaslr_get_random_long(purpose: *const i8) -> u64 {
let mut mix_const: u64 = 6728387515348454867;
let mut raw: u64 = 0;
let mut random: u64 = get_boot_seed();
let mut use_i8254: bool = true;
if !purpose.is_null() {
debug_putstr__3!(purpose);
debug_putstr!(c" KASLR using".as_ptr() as *mut i8);
}
if unsafe { has_cpuflag(158i32) } {
if !purpose.is_null() {
debug_putstr!(c" RDRAND".as_ptr() as *mut i8);
}
if rdrand_long(unsafe { &mut (*core::ptr::addr_of_mut!(raw)) }) {
random ^= raw;
use_i8254 = false;
}
}
if unsafe { has_cpuflag(4i32) } {
if !purpose.is_null() {
debug_putstr!(c" RDTSC".as_ptr() as *mut i8);
}
raw = rdtsc();
random ^= raw;
use_i8254 = false;
}
if use_i8254 {
if !purpose.is_null() {
debug_putstr!(c" i8254".as_ptr() as *mut i8);
}
let __v4: u16 = i8254();
random ^= (__v4 as i32) as u64;
}
// Circular multiply for better bit diffusion
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!(" mulq {0:r}", in(reg) mix_const as u64, lateout("ax") __t0, lateout("dx") __t1, in("ax") random as u64, options(pure, nomem, nostack, att_syntax));
random = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
raw = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
random = random.wrapping_add(raw);
if !purpose.is_null() {
debug_putstr!(c"...\n".as_ptr() as *mut i8);
}
random
}

// Only supporting at most 4 unusable memmap regions with kaslr
static mut memmap_too_large: bool = unsafe { core::mem::zeroed() };

/// Store memory limit: MAXMEM on 64-bit and KERNEL_IMAGE_SIZE on 32-bit.
/// It may be reduced by "mem=nn[KMG]" or "memmap=nn[KMG]" command line options.
static mut mem_limit: u64 = unsafe { core::mem::zeroed() };

/// Number of immovable memory regions
static mut num_immovable_mem: i32 = unsafe { core::mem::zeroed() };

static mut mem_avoid: __SlateAlign16<[mem_vector; 8]> = __SlateAlign16(unsafe { core::mem::zeroed() });

fn mem_overlaps(one: &mem_vector, two: *mut mem_vector) -> bool {
// Item one is entirely before item two.
if unsafe { (*(one as *const mem_vector)).start }.wrapping_add(unsafe { (*(one as *const mem_vector)).size }) <= unsafe { (*two).start } {
return false;
}
// Item one is entirely after item two.
if (unsafe { (*(one as *const mem_vector)).start }) >= unsafe { (*two).start }.wrapping_add(unsafe { (*two).size }) {
return false;
}
true
}

#[unsafe(no_mangle)]
extern "C" fn skip_spaces(mut str: *const i8) -> *mut i8 {
while isspace!(unsafe { *str }) {
let __v0: *const i8 = str;
str = unsafe { __v0.add(1) };
}
str as *mut i8
}

#[unsafe(no_mangle)]
static _ctype: __SlateAlign16<[u8; 256]> = __SlateAlign16([8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 40u8, 40u8, 40u8, 40u8, 40u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 8u8, 160u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 4u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 66u8, 66u8, 66u8, 66u8, 66u8, 66u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 16u8, 16u8, 16u8, 16u8, 8u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 160u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 16u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 1u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 16u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8, 2u8]);

fn get_range(str: &mut *mut i8, mut pint: *mut i32, mut n: i32) -> i32 {
let mut x: i32 = 0;
let mut inc_counter: i32 = 0;
let mut upper_range: i32 = 0;
let __v1: *mut i8 = unsafe { *str };
unsafe {
*str = __v1.add(1);
}
upper_range = (unsafe { simple_strtol(*(str as *mut *mut i8) as *const i8, core::ptr::null_mut::<*mut i8>(), 0u32) }) as i32;
inc_counter = upper_range - unsafe { *pint };
x = unsafe { *pint };
while n != 0 && x < upper_range {
let __v3: *mut i32 = pint;
pint = unsafe { __v3.add(1) };
unsafe {
*__v3 = x;
}
x += 1;
n -= 1;
}
inc_counter
}

#[unsafe(no_mangle)]
extern "C" fn get_option(str: *mut *mut i8, pint: *mut i32) -> i32 {
let mut cur: *mut i8 = unsafe { *str };
let mut value: i32 = 0;
if cur.is_null() || (unsafe { *cur }) == 0 {
return 0;
}
if ((unsafe { *cur }) as i32) == 45 {
let __v0: *mut i8 = cur;
let __v1: *mut i8 = unsafe { __v0.add(1) };
cur = __v1;
value = unsafe { simple_strtoull(__v1 as *const i8, str, 0u32) }.wrapping_neg() as i32;
} else {
value = (unsafe { simple_strtoull(cur as *const i8, str, 0u32) }) as i32;
}
if !pint.is_null() {
unsafe {
*pint = value;
}
}
if cur == unsafe { *str } {
return 0;
}
if ((unsafe { *(*str) }) as i32) == 44 {
let __v3: *mut i8 = unsafe { *str };
unsafe {
*str = __v3.add(1);
}
return 2;
}
if ((unsafe { *(*str) }) as i32) == 45 {
return 3;
}
1
}

#[unsafe(no_mangle)]
extern "C" fn get_options(mut str: *const i8, nints: i32, ints: *mut i32) -> *mut i8 {
let validate: bool = nints == 0;
let mut res: i32 = 0;
let mut i__0: i32 = 1;
while i__0 < nints || validate {
let pint: *mut i32 = if validate { ints } else { unsafe { ints.offset(i__0 as isize) } };
res = get_option(core::ptr::addr_of_mut!(str) as *mut *mut i8, pint);
if res == 0 {
break;
}
if res == 3 {
let n: i32 = if validate { 0 } else { nints - i__0 };
let range_nums: i32 = get_range(unsafe { &mut (*(core::ptr::addr_of_mut!(str) as *mut *mut i8)) }, pint, n);
if range_nums < 0 {
break;
}
// Decrement the result by one to leave out the
// last number in the range.  The next iteration
// will handle the upper number in the range
i__0 += range_nums - 1;
}
i__0 += 1;
if res == 1 {
break;
}
}
unsafe {
*ints.add(0) = i__0 - 1;
}
str as *mut i8
}

#[unsafe(no_mangle)]
extern "C" fn memparse(ptr: *const i8, retptr: *mut *mut i8) -> u64 {
let mut endptr: *mut i8 = unsafe { core::mem::zeroed() }; // local pointer to end of parsed string
let mut ret: u64 = unsafe { simple_strtoull(ptr, core::ptr::addr_of_mut!(endptr), 0u32) };
let mut shl: u32 = 0;
// Consume valid suffix even in case of overflow.
let mut __t0: i64 = match (unsafe { *endptr }) as i32 { 69 | 101 => 0, 80 | 112 => 1, 84 | 116 => 2, 71 | 103 => 3, 77 | 109 => 4, 75 | 107 => 5, _ => 6 };
loop {
match __t0 {
0 => {
shl = shl.wrapping_add(10);
__t0 = 1;
continue;
}
1 => {
shl = shl.wrapping_add(10);
__t0 = 2;
continue;
}
2 => {
shl = shl.wrapping_add(10);
__t0 = 3;
continue;
}
3 => {
shl = shl.wrapping_add(10);
__t0 = 4;
continue;
}
4 => {
shl = shl.wrapping_add(10);
__t0 = 5;
continue;
}
5 => {
shl = shl.wrapping_add(10);
__t0 = 6;
continue;
}
6 => {
break;
}
_ => {
break;
}
}
}
if shl != 0 && likely!(ptr != (endptr as *const i8)) != 0 {
// Have valid suffix with preceding number.
if unlikely!(check_shl_overflow!(ret, shl, core::ptr::addr_of_mut!(ret))) != 0 {
ret = !0;
}
let __v19: *mut i8 = endptr;
endptr = unsafe { __v19.add(1) };
}
if !retptr.is_null() {
unsafe {
*retptr = endptr;
}
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn parse_option_str(mut str: *const i8, option: *const i8) -> bool {
while (unsafe { *str }) != 0 {
if (unsafe { strncmp(str, option, strlen(option)) }) == 0 {
let __v0: u64 = unsafe { strlen(option) };
let __v1: *const i8 = str;
str = unsafe { __v1.add(__v0 as usize) };
if (unsafe { *str }) == 0 || ((unsafe { *str }) as i32) == 44 {
return true;
}
}
while (unsafe { *str }) != 0 && ((unsafe { *str }) as i32) != 44 {
let __v3: *const i8 = str;
str = unsafe { __v3.add(1) };
}
if ((unsafe { *str }) as i32) == 44 {
let __v6: *const i8 = str;
str = unsafe { __v6.add(1) };
}
}
false
}

#[unsafe(no_mangle)]
extern "C" fn next_arg(mut args: *mut i8, param: *mut *mut i8, val: *mut *mut i8) -> *mut i8 {
let mut i__0: u32 = 0;
let mut equals: u32 = 0;
let mut in_quote: i32 = 0;
let mut quoted: i32 = 0;
if ((unsafe { *args }) as i32) == 34 {
let __v0: *mut i8 = args;
args = unsafe { __v0.add(1) };
in_quote = 1;
quoted = 1;
}
while (unsafe { *args.add(i__0 as usize) }) != 0 {
if isspace__2!(unsafe { *args.add(i__0 as usize) }) && in_quote == 0 {
break;
}
if equals == 0 {
if ((unsafe { *args.add(i__0 as usize) }) as i32) == 61 {
equals = i__0;
}
}
if ((unsafe { *args.add(i__0 as usize) }) as i32) == 34 {
in_quote = (in_quote == 0) as i32;
}
i__0 = i__0.wrapping_add(1);
}
unsafe {
*param = args;
}
if equals == 0 {
unsafe {
*val = core::ptr::null_mut::<i8>();
}
} else {
unsafe {
*args.add(equals as usize) = 0;
}
unsafe {
*val = args.add(equals as usize).add(1);
}
// Don't include quotes in value.
if ((unsafe { *(*val) }) as i32) == 34 {
let __v6: *mut i8 = unsafe { *val };
unsafe {
*val = __v6.add(1);
}
if ((unsafe { *args.offset(i__0.wrapping_sub(1) as isize) }) as i32) == 34 {
unsafe {
*args.offset(i__0.wrapping_sub(1) as isize) = 0;
}
}
}
}
if quoted != 0 && i__0 > 0 && ((unsafe { *args.offset(i__0.wrapping_sub(1) as isize) }) as i32) == 34 {
unsafe {
*args.offset(i__0.wrapping_sub(1) as isize) = 0;
}
}
if (unsafe { *args.add(i__0 as usize) }) != 0 {
unsafe {
*args.add(i__0 as usize) = 0;
}
let __v8: *mut i8 = args;
args = unsafe { __v8.offset(i__0.wrapping_add(1) as isize) };
} else {
let __v10: *mut i8 = args;
args = unsafe { __v10.add(i__0 as usize) };
}
// Chew up trailing spaces.
skip_spaces(args as *const i8)
}

fn parse_memmap(mut p: *mut i8, start: &mut u64, size: &mut u64) -> i32 {
if p.is_null() {
return -22;
}
// We don't care about this option here
if (unsafe { strncmp(p as *const i8, c"exactmap".as_ptr(), 8u64) }) == 0 {
return -22;
}
let oldp: *mut i8 = p;
unsafe {
*(size as *mut u64) = memparse(p as *const i8, core::ptr::addr_of_mut!(p));
}
if p == oldp {
return -22;
}
// memmap=nn@ss specifies usable region, should
// be skipped
// If w/o offset, only size specified, memmap=nn[KMG] has the
// same behaviour as mem=nn[KMG]. It limits the max address
// system can use. Region above the limit should be avoided.
match (unsafe { *p }) as i32 {
33 | 35 | 36 => {
unsafe {
*(start as *mut u64) = memparse(p.add(1) as *const i8, core::ptr::addr_of_mut!(p));
}
return 0;
}
64 => {
unsafe {
*(size as *mut u64) = 0;
}
// memmap=nn@ss specifies usable region, should
// be skipped
unsafe {
*(start as *mut u64) = 0;
}
// If w/o offset, only size specified, memmap=nn[KMG] has the
// same behaviour as mem=nn[KMG]. It limits the max address
// system can use. Region above the limit should be avoided.
return 0;
}
_ => {
unsafe {
*(start as *mut u64) = 0;
}
// If w/o offset, only size specified, memmap=nn[KMG] has the
// same behaviour as mem=nn[KMG]. It limits the max address
// system can use. Region above the limit should be avoided.
return 0;
}
}
-22
}

fn mem_avoid_memmap(mut str: *mut i8) {
if (unsafe { i }) >= 4 {
return;
}
while !str.is_null() && (unsafe { i }) < 4 {
let mut rc: i32 = 0;
let mut start: u64 = 0;
let mut size: u64 = 0;
let mut k: *mut i8 = unsafe { strchr(str as *const i8, 44i32) };
if !k.is_null() {
let __v0: *mut i8 = k;
k = unsafe { __v0.add(1) };
unsafe {
*__v0 = 0;
}
}
rc = parse_memmap(str, unsafe { &mut (*core::ptr::addr_of_mut!(start)) }, unsafe { &mut (*core::ptr::addr_of_mut!(size)) });
if rc < 0 {
break;
}
str = k;
if start == 0 {
// Store the specified memory limit if size > 0
if size > 0 && size < unsafe { mem_limit } {
unsafe {
mem_limit = size;
}
}
} else {
unsafe {
mem_avoid.0[(4 + i) as usize].start = start;
}
unsafe {
mem_avoid.0[(4 + i) as usize].size = size;
}
unsafe {
i = i + 1;
}
}
}
// More than 4 memmaps, fail kaslr
if (unsafe { i }) >= 4 && !str.is_null() {
unsafe {
memmap_too_large = true;
}
}
}

static mut i: i32 = unsafe { core::mem::zeroed() };

/// Store the number of 1GB huge pages which users specified:
static mut max_gb_huge_pages: u64 = unsafe { core::mem::zeroed() };

fn parse_gb_huge_pages(param: *mut i8, val: *mut i8) {
let mut p: *mut i8 = unsafe { core::mem::zeroed() };
if (unsafe { strcmp(param as *const i8, c"hugepagesz".as_ptr()) }) == 0 {
p = val;
if memparse(p as *const i8, core::ptr::addr_of_mut!(p)) != _AC!() << 30i32 {
unsafe {
gbpage_sz = false;
}
return;
}
if unsafe { gbpage_sz } {
unsafe { warn(c"Repeatedly set hugeTLB page size of 1G!\n".as_ptr()) };
}
unsafe {
gbpage_sz = true;
}
return;
}
if (unsafe { strcmp(param as *const i8, c"hugepages".as_ptr()) }) == 0 && unsafe { gbpage_sz } {
p = val;
if (unsafe { boot_kstrtoul(p as *const i8, 0u32, core::ptr::addr_of_mut!(max_gb_huge_pages)) }) != 0 {
unsafe { warn(c"Failed to parse hugepages= boot parameter\n".as_ptr()) };
}
return;
}
}

static mut gbpage_sz: bool = unsafe { core::mem::zeroed() };

fn handle_mem_options() {
let mut args: *mut i8 = (unsafe { get_cmd_line_ptr() }) as *mut i8;
let mut len: u64 = 0;
let mut param: *mut i8 = unsafe { core::mem::zeroed() };
let mut val: *mut i8 = unsafe { core::mem::zeroed() };
let mut mem_size: u64 = 0;
if args.is_null() {
return;
}
len = unsafe { strnlen(args as *const i8, 2047u64) };
let tmp_cmdline: *mut i8 = (unsafe { malloc(len.wrapping_add(1) as i32) }) as *mut i8;
if tmp_cmdline.is_null() {
unsafe { error(c"Failed to allocate space for tmp_cmdline".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
memcpy!(tmp_cmdline, args, len);
unsafe {
*tmp_cmdline.add(len as usize) = 0;
}
args = tmp_cmdline;
// Chew leading spaces
args = skip_spaces(args as *const i8);
while (unsafe { *args }) != 0 {
args = next_arg(args, core::ptr::addr_of_mut!(param), core::ptr::addr_of_mut!(val));
// Stop at --
let __v0: bool = val.is_null() && (unsafe { strcmp(param as *const i8, c"--".as_ptr()) }) == 0;
if __v0 {
break;
}
if (unsafe { strcmp(param as *const i8, c"memmap".as_ptr()) }) == 0 {
mem_avoid_memmap(val);
} else {
let __v2: bool = IS_ENABLED__2!() != 0 && !unsafe { strstr(param as *const i8, c"hugepages".as_ptr()) }.is_null();
if __v2 {
parse_gb_huge_pages(param, val);
} else {
if (unsafe { strcmp(param as *const i8, c"mem".as_ptr()) }) == 0 {
let mut p: *mut i8 = val;
if (unsafe { strcmp(p as *const i8, c"nopentium".as_ptr()) }) != 0 {
mem_size = memparse(p as *const i8, core::ptr::addr_of_mut!(p));
if mem_size == 0 {
break;
}
if mem_size < unsafe { mem_limit } {
unsafe {
mem_limit = mem_size;
}
}
}
}
}
}
}
unsafe { free(tmp_cmdline as *mut ()) };
return;
}

/// In theory, KASLR can put the kernel anywhere in the range of [16M, MAXMEM)
/// on 64-bit, and [16M, KERNEL_IMAGE_SIZE) on 32-bit.
///
/// The mem_avoid array is used to store the ranges that need to be avoided
/// when KASLR searches for an appropriate random address. We must avoid any
/// regions that are unsafe to overlap with during decompression, and other
/// things like the initrd, cmdline and boot_params. This comment seeks to
/// explain mem_avoid as clearly as possible since incorrect mem_avoid
/// memory ranges lead to really hard to debug boot failures.
///
/// The initrd, cmdline, and boot_params are trivial to identify for
/// avoiding. They are MEM_AVOID_INITRD, MEM_AVOID_CMDLINE, and
/// MEM_AVOID_BOOTPARAMS respectively below.
///
/// What is not obvious how to avoid is the range of memory that is used
/// during decompression (MEM_AVOID_ZO_RANGE below). This range must cover
/// the compressed kernel (ZO) and its run space, which is used to extract
/// the uncompressed kernel (VO) and relocs.
///
/// ZO's full run size sits against the end of the decompression buffer, so
/// we can calculate where text, data, bss, etc of ZO are positioned more
/// easily.
///
/// For additional background, the decompression calculations can be found
/// in header.S, and the memory diagram is based on the one found in misc.c.
///
/// The following conditions are already enforced by the image layouts and
/// associated code:
///  - input + input_size >= output + output_size
///  - kernel_total_size <= init_size
///  - kernel_total_size <= output_size (see Note below)
///  - output + init_size >= output + output_size
///
/// (Note that kernel_total_size and output_size have no fundamental
/// relationship, but output_size is passed to choose_random_location
/// as a maximum of the two. The diagram is showing a case where
/// kernel_total_size is larger than output_size, but this case is
/// handled by bumping output_size.)
///
/// The above conditions can be illustrated by a diagram:
///
/// 0   output            input            input+input_size    output+init_size
/// |     |                 |                             |             |
/// |     |                 |                             |             |
/// |-----|--------|--------|--------------|-----------|--|-------------|
///                |                       |           |
///                |                       |           |
/// output+init_size-ZO_INIT_SIZE  output+output_size  output+kernel_total_size
///
/// [output, output+init_size) is the entire memory range used for
/// extracting the compressed image.
///
/// [output, output+kernel_total_size) is the range needed for the
/// uncompressed kernel (VO) and its run size (bss, brk, etc).
///
/// [output, output+output_size) is VO plus relocs (i.e. the entire
/// uncompressed payload contained by ZO). This is the area of the buffer
/// written to during decompression.
///
/// [output+init_size-ZO_INIT_SIZE, output+init_size) is the worst-case
/// range of the copied ZO and decompression code. (i.e. the range
/// covered backwards of size ZO_INIT_SIZE, starting from output+init_size.)
///
/// [input, input+input_size) is the original copied compressed image (ZO)
/// (i.e. it does not include its run size). This range must be avoided
/// because it contains the data used for decompression.
///
/// [input+input_size, output+init_size) is [_text, _end) for ZO. This
/// range includes ZO's heap and stack, and must be avoided since it
/// performs the decompression.
///
/// Since the above two ranges need to be avoided and they are adjacent,
/// they can be merged, resulting in: [input, output+init_size) which
/// becomes the MEM_AVOID_ZO_RANGE below.
fn mem_avoid_init(input: u64, input_size: u64, output: u64) {
let init_size: u64 = (unsafe { (*boot_params_ptr).hdr.init_size }) as u64;
let mut initrd_start: u64 = 0;
let mut initrd_size: u64 = 0;
let mut cmd_line: u64 = 0;
let mut cmd_line_size: u64 = 0;
// Avoid the region that is unsafe to overlap during
// decompression.
unsafe {
mem_avoid.0[0].start = input;
}
unsafe {
mem_avoid.0[0].size = output.wrapping_add(init_size).wrapping_sub(input);
}
// Avoid initrd.
initrd_start = ((unsafe { (*boot_params_ptr).ext_ramdisk_image }) as u64) << 32i32;
initrd_start |= (unsafe { (*boot_params_ptr).hdr.ramdisk_image }) as u64;
initrd_size = ((unsafe { (*boot_params_ptr).ext_ramdisk_size }) as u64) << 32i32;
initrd_size |= (unsafe { (*boot_params_ptr).hdr.ramdisk_size }) as u64;
unsafe {
mem_avoid.0[1].start = initrd_start;
}
unsafe {
mem_avoid.0[1].size = initrd_size;
}
// No need to set mapping for initrd, it will be handled in VO.
// Avoid kernel command line.
cmd_line = unsafe { get_cmd_line_ptr() };
// Calculate size of cmd_line.
if cmd_line != 0 {
cmd_line_size = unsafe { strnlen(cmd_line as *const i8, 2047u64) }.wrapping_add(1);
unsafe {
mem_avoid.0[2].start = cmd_line;
}
unsafe {
mem_avoid.0[2].size = cmd_line_size;
}
}
// Avoid boot parameters.
unsafe {
mem_avoid.0[3].start = boot_params_ptr as u64;
}
unsafe {
mem_avoid.0[3].size = core::mem::size_of::<boot_params>() as u64;
}
// We don't need to set a mapping for setup_data.
// Mark the memmap regions we need to avoid
handle_mem_options();
// Enumerate the immovable memory regions
unsafe {
num_immovable_mem = count_immovable_mem_regions();
}
}

/// Does this memory vector overlap a known avoided area? If so, record the
/// overlap region with the lowest address.
fn mem_avoid_overlap(img: &mem_vector, overlap: &mut mem_vector) -> bool {
let mut ptr: *mut setup_data = unsafe { core::mem::zeroed() };
let mut earliest: u64 = unsafe { (*(img as *const mem_vector)).start }.wrapping_add(unsafe { (*(img as *const mem_vector)).size });
let mut is_overlapping: bool = false;
for i__0 in 0i32..8i32 {
if mem_overlaps(img, unsafe { (core::ptr::addr_of_mut!(mem_avoid.0) as *mut mem_vector).offset(i__0 as isize) }) && (unsafe { mem_avoid.0[i__0 as usize].start }) < earliest {
unsafe {
*(overlap as *mut mem_vector) = mem_avoid.0[i__0 as usize];
}
earliest = unsafe { (*(overlap as *mut mem_vector)).start };
is_overlapping = true;
}
}
// Avoid all entries in the setup_data linked list.
ptr = (unsafe { (*boot_params_ptr).hdr.setup_data }) as *mut setup_data;
while !ptr.is_null() {
let mut avoid: mem_vector = unsafe { core::mem::zeroed() };
avoid.start = ptr as u64;
avoid.size = 16u64.wrapping_add((unsafe { (*ptr).len }) as u64);
if mem_overlaps(img, core::ptr::addr_of_mut!(avoid)) && avoid.start < earliest {
unsafe {
*(overlap as *mut mem_vector) = avoid;
}
earliest = unsafe { (*(overlap as *mut mem_vector)).start };
is_overlapping = true;
}
if (unsafe { (*ptr).r#type }) == 2147483648u32 && (unsafe { (*((core::ptr::addr_of_mut!((*ptr).data) as *mut u8) as *mut setup_indirect)).r#type }) != 2147483648u32 {
avoid.start = unsafe { (*((core::ptr::addr_of_mut!((*ptr).data) as *mut u8) as *mut setup_indirect)).addr };
avoid.size = unsafe { (*((core::ptr::addr_of_mut!((*ptr).data) as *mut u8) as *mut setup_indirect)).len };
if mem_overlaps(img, core::ptr::addr_of_mut!(avoid)) && avoid.start < earliest {
unsafe {
*(overlap as *mut mem_vector) = avoid;
}
earliest = unsafe { (*(overlap as *mut mem_vector)).start };
is_overlapping = true;
}
}
ptr = (unsafe { (*ptr).next }) as *mut setup_data;
}
is_overlapping
}

#[repr(C)]
#[derive(Clone, Copy)]
struct slot_area {
    addr: u64,
    num: u64,
}


static mut slot_areas: __SlateAlign16<[slot_area; 100]> = __SlateAlign16(unsafe { core::mem::zeroed() });

static mut slot_area_index: u32 = unsafe { core::mem::zeroed() };

static mut slot_max: u64 = unsafe { core::mem::zeroed() };

fn store_slot_info(region: *mut mem_vector, image_size: u64) {
let mut slot_area: slot_area = unsafe { core::mem::zeroed() };
if (unsafe { slot_area_index }) == 100 {
return;
}
slot_area.addr = unsafe { (*region).start };
slot_area.num = 1u64.wrapping_add(unsafe { (*region).size }.wrapping_sub(image_size) / 2097152);
let __v0: u32 = unsafe { slot_area_index };
unsafe {
slot_area_index = __v0.wrapping_add(1);
}
unsafe {
slot_areas.0[__v0 as usize] = slot_area;
}
let __v2: u64 = unsafe { slot_max };
unsafe {
slot_max = __v2.wrapping_add(slot_area.num);
}
}

/// Skip as many 1GB huge pages as possible in the passed region
/// according to the number which users specified:
fn process_gb_huge_pages(region: *mut mem_vector, image_size: u64) {
let mut pud_start: u64 = 0;
let mut pud_end: u64 = 0;
let mut gb_huge_pages: u64 = 0;
let mut tmp: mem_vector = unsafe { core::mem::zeroed() };
if IS_ENABLED__2!() == 0 || (unsafe { max_gb_huge_pages }) == 0 {
store_slot_info(region, image_size);
return;
}
// Are there any 1GB pages in the region?
pud_start = ALIGN!(unsafe { (*region).start }, 1073741824u64);
pud_end = ALIGN_DOWN!(unsafe { (*region).start }.wrapping_add(unsafe { (*region).size }), 1073741824u64);
// No good 1GB huge pages found:
if pud_start >= pud_end {
store_slot_info(region, image_size);
return;
}
// Check if the head part of the region is usable.
if pud_start >= unsafe { (*region).start }.wrapping_add(image_size) {
tmp.start = unsafe { (*region).start };
tmp.size = pud_start.wrapping_sub(unsafe { (*region).start });
store_slot_info(core::ptr::addr_of_mut!(tmp), image_size);
}
// Skip the good 1GB pages.
gb_huge_pages = pud_end.wrapping_sub(pud_start) >> 30i32;
if gb_huge_pages > unsafe { max_gb_huge_pages } {
pud_end = pud_start.wrapping_add((unsafe { max_gb_huge_pages }) << 30i32);
unsafe {
max_gb_huge_pages = 0;
}
} else {
let __v0: u64 = unsafe { max_gb_huge_pages };
unsafe {
max_gb_huge_pages = __v0.wrapping_sub(gb_huge_pages);
}
}
// Check if the tail part of the region is usable.
if unsafe { (*region).start }.wrapping_add(unsafe { (*region).size }) >= pud_end.wrapping_add(image_size) {
tmp.start = pud_end;
tmp.size = unsafe { (*region).start }.wrapping_add(unsafe { (*region).size }).wrapping_sub(pud_end);
store_slot_info(core::ptr::addr_of_mut!(tmp), image_size);
}
}

fn slots_fetch_random() -> u64 {
let mut slot: u64 = 0;
let mut i__0: u32 = 0;
// Handle case of no slots stored.
if (unsafe { slot_max }) == 0 {
return 0;
}
slot = kaslr_get_random_long(c"Physical".as_ptr()) % unsafe { slot_max };
while i__0 < unsafe { slot_area_index } {
if slot >= unsafe { slot_areas.0[i__0 as usize].num } {
slot = slot.wrapping_sub(unsafe { slot_areas.0[i__0 as usize].num });
} else {
return unsafe { slot_areas.0[i__0 as usize].addr }.wrapping_add(slot.wrapping_mul(2097152));
}
i__0 = i__0.wrapping_add(1);
}
if i__0 == unsafe { slot_area_index } {
debug_putstr!(c"slots_fetch_random() failed!?\n".as_ptr() as *mut i8);
}
0
}

fn __process_mem_region(entry: *mut mem_vector, minimum: u64, image_size: u64) {
let mut region: mem_vector = unsafe { core::mem::zeroed() };
let mut overlap: mem_vector = unsafe { core::mem::zeroed() };
let mut region_end: u64 = 0;
// Enforce minimum and memory limit.
region.start = max_t__2!(u64, unsafe { (*entry).start }, minimum);
let mut __v3: u64;
let __UNIQUE_ID_x__463: u64 = unsafe { (*entry).start }.wrapping_add(unsafe { (*entry).size });
let __UNIQUE_ID_y__464: u64 = unsafe { mem_limit };
loop {
if __types_ok!(__UNIQUE_ID_x__463, __UNIQUE_ID_y__464) == 0 {
unsafe { __compiletime_assert_465() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__463 < __UNIQUE_ID_y__464 { __UNIQUE_ID_x__463 } else { __UNIQUE_ID_y__464 };
region_end = __v3;
// Give up if slot area array is full.
while (unsafe { slot_area_index }) < 100 {
// Potentially raise address to meet alignment needs.
region.start = ALIGN!(region.start, 2097152i32);
// Did we raise the address above the passed in memory entry?
if region.start > region_end {
return;
}
// Reduce size by any delta from the original address.
region.size = region_end.wrapping_sub(region.start);
// Return if region can't contain decompressed kernel
if region.size < image_size {
return;
}
// If nothing overlaps, store the region and return.
if !mem_avoid_overlap(unsafe { &(*core::ptr::addr_of_mut!(region)) }, unsafe { &mut (*core::ptr::addr_of_mut!(overlap)) }) {
process_gb_huge_pages(core::ptr::addr_of_mut!(region), image_size);
return;
}
// Store beginning of region if holds at least image_size.
if overlap.start >= region.start.wrapping_add(image_size) {
region.size = overlap.start.wrapping_sub(region.start);
process_gb_huge_pages(core::ptr::addr_of_mut!(region), image_size);
}
// Clip off the overlapping region and start over.
region.start = overlap.start.wrapping_add(overlap.size);
}
}

fn process_mem_region(region: *mut mem_vector, minimum: u64, image_size: u64) -> bool {
// If no immovable memory found, or MEMORY_HOTREMOVE disabled,
// use @region directly.
if (unsafe { num_immovable_mem }) == 0 {
__process_mem_region(region, minimum, image_size);
if (unsafe { slot_area_index }) == 100 {
debug_putstr!(c"Aborted e820/efi memmap scan (slot_areas full)!\n".as_ptr() as *mut i8);
return true;
}
return false;
}
false
}

/// Only EFI_CONVENTIONAL_MEMORY and EFI_UNACCEPTED_MEMORY (if supported) are
/// guaranteed to be free.
///
/// Pick free memory more conservatively than the EFI spec allows: according to
/// the spec, EFI_BOOT_SERVICES_{CODE|DATA} are also free memory and thus
/// available to place the kernel image into, but in practice there's firmware
/// where using that memory leads to crashes. Buggy vendor EFI code registers
/// for an event that triggers on SetVirtualAddressMap(). The handler assumes
/// that EFI_BOOT_SERVICES_DATA memory has not been touched by loader yet, which
/// is probably true for Windows.
///
/// Preserve EFI_BOOT_SERVICES_* regions until after SetVirtualAddressMap().
fn memory_type_is_free(md: *mut __SlateRecord_efi_h_51) -> bool {
if (unsafe { (*md).r#type }) == 7 {
return true;
}
if IS_ENABLED!() != 0 && (unsafe { (*md).r#type }) == 15 {
return true;
}
false
}

/// Returns true if we processed the EFI memmap, which we prefer over the E820
/// table if it is available.
fn process_efi_entries(minimum: u64, image_size: u64) -> bool {
let e: *mut efi_info = unsafe { core::ptr::addr_of_mut!((*boot_params_ptr).efi_info) };
let mut efi_mirror_found: bool = false;
let mut region: mem_vector = unsafe { core::mem::zeroed() };
let mut md: *mut __SlateRecord_efi_h_51 = unsafe { core::mem::zeroed() };
let mut pmap: u64 = 0;
let mut nr_desc: u32 = 0;
let mut i__0: i32 = 0;
let signature: *mut i8 = (unsafe { core::ptr::addr_of_mut!((*e).efi_loader_signature) }) as *mut i8;
let __v0: bool = (unsafe { strncmp(signature as *const i8, c"EL32".as_ptr(), 4u64) }) != 0 && (unsafe { strncmp(signature as *const i8, c"EL64".as_ptr(), 4u64) }) != 0;
if __v0 {
return false;
}
pmap = ((unsafe { (*e).efi_memmap }) as u64) | ((unsafe { (*e).efi_memmap_hi }) as u64) << 32i32;
nr_desc = (unsafe { (*e).efi_memmap_size }) / unsafe { (*e).efi_memdesc_size };
while (i__0 as u32) < nr_desc {
md = efi_early_memdesc_ptr!(pmap, unsafe { (*e).efi_memdesc_size }, i__0);
if (unsafe { (*md).attribute }) & 65536 != 0 {
efi_mirror_found = true;
break;
}
i__0 += 1;
}
i__0 = 0;
while (i__0 as u32) < nr_desc {
md = efi_early_memdesc_ptr!(pmap, unsafe { (*e).efi_memdesc_size }, i__0);
if memory_type_is_free(md) {
if !(efi_soft_reserve_enabled() && (unsafe { (*md).attribute }) & 262144 != 0) {
if !(efi_mirror_found && (unsafe { (*md).attribute }) & 65536 == 0) {
region.start = unsafe { (*md).phys_addr };
region.size = (unsafe { (*md).num_pages }) << 12i32;
if process_mem_region(core::ptr::addr_of_mut!(region), minimum, image_size) {
break;
}
}
}
}
i__0 += 1;
}
true
}

fn process_e820_entries(minimum: u64, image_size: u64) {
let mut i__0: i32 = 0;
let mut region: mem_vector = unsafe { core::mem::zeroed() };
// Verify potential e820 positions, appending to slots list.
while i__0 < ((unsafe { (*boot_params_ptr).e820_entries }) as i32) {
let entry: *mut boot_e820_entry = unsafe { (core::ptr::addr_of_mut!((*boot_params_ptr).e820_table) as *mut boot_e820_entry).offset(i__0 as isize) };
// Skip non-RAM entries.
if (unsafe { (*entry).r#type }) == 1 {
region.start = unsafe { (*entry).addr };
region.size = unsafe { (*entry).size };
if process_mem_region(core::ptr::addr_of_mut!(region), minimum, image_size) {
break;
}
}
i__0 += 1;
}
}

/// If KHO is active, only process its scratch areas to ensure we are not
/// stepping onto preserved memory.
fn process_kho_entries(minimum: u64, image_size: u64) -> bool {
let mut kho_scratch: *mut kho_scratch = unsafe { core::mem::zeroed() };
let mut ptr: *mut setup_data = unsafe { core::mem::zeroed() };
let mut nr_areas: i32 = 0;
if IS_ENABLED!() == 0 {
return false;
}
ptr = (unsafe { (*boot_params_ptr).hdr.setup_data }) as *mut setup_data;
while !ptr.is_null() {
if (unsafe { (*ptr).r#type }) == 10 {
let kho: *mut kho_data = ((unsafe { core::ptr::addr_of_mut!((*ptr).data) as *mut u8 }) as u64) as *mut kho_data;
kho_scratch = ((unsafe { (*kho).scratch_addr }) as *mut ()) as *mut kho_scratch;
nr_areas = ((unsafe { (*kho).scratch_size }) / (core::mem::size_of::<kho_scratch>() as u64)) as i32;
break;
}
ptr = (unsafe { (*ptr).next }) as *mut setup_data;
}
if nr_areas == 0 {
return false;
}
for i__0 in 0..nr_areas {
let area: *mut kho_scratch = unsafe { kho_scratch.offset(i__0 as isize) };
let mut region: mem_vector = mem_vector { start: unsafe { (*area).addr }, size: unsafe { (*area).size } };
if process_mem_region(core::ptr::addr_of_mut!(region), minimum, image_size) {
break;
}
}
true
}

fn find_random_phys_addr(minimum: u64, image_size: u64) -> u64 {
let mut phys_addr: u64 = 0;
// Bail out early if it's impossible to succeed.
if minimum.wrapping_add(image_size) > unsafe { mem_limit } {
return 0;
}
// Check if we had too many memmaps.
if unsafe { memmap_too_large } {
debug_putstr!(c"Aborted memory entries scan (more than 4 memmap= args)!\n".as_ptr() as *mut i8);
return 0;
}
// During kexec handover only process KHO scratch areas that are known
// not to contain any data that must be preserved.
let __v0: bool = !process_kho_entries(minimum, image_size) && !process_efi_entries(minimum, image_size);
if __v0 {
process_e820_entries(minimum, image_size);
}
phys_addr = slots_fetch_random();
// Perform a final check to make sure the address is in range.
if phys_addr < minimum || phys_addr.wrapping_add(image_size) > unsafe { mem_limit } {
unsafe { warn(c"Invalid physical address chosen!\n".as_ptr()) };
return 0;
}
phys_addr
}

fn find_random_virt_addr(minimum: u64, image_size: u64) -> u64 {
let mut slots: u64 = 0;
let mut random_addr: u64 = 0;
// There are how many CONFIG_PHYSICAL_ALIGN-sized slots
// that can hold image_size within the range of minimum to
// KERNEL_IMAGE_SIZE?
slots = 1u64.wrapping_add(1073741824u64.wrapping_sub(minimum).wrapping_sub(image_size) / 2097152);
random_addr = kaslr_get_random_long(c"Virtual".as_ptr()) % slots;
random_addr.wrapping_mul(2097152).wrapping_add(minimum)
}

// Since this function examines addresses much more numerically,
// it takes the input and output pointers as 'unsigned long'.
