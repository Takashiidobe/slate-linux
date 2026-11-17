
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
//! Copyright (C) 2014 Red Hat
//! Copyright (C) 2014 Intel Corp.
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sublicense,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
//! THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
//! OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
//! ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
//! OTHER DEALINGS IN THE SOFTWARE.
//!
//! Authors:
//! Rob Clark <robdclark@gmail.com>
//! Daniel Vetter <daniel.vetter@ffwll.ch>
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
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
//! Sign Flag
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
//! @a is a power of 2 value
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
//! for_each_set_bit_wrap - iterate over all set bits starting from @start, and
//! wrapping around the end of bitmap.
//! @bit: offset for current iteration
//! @addr: bitmap address to base the search on
//! @size: bitmap size in number of bits
//! @start: Starting bit for bitmap traversing, wrapping around the bitmap end
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! do_div - returns 2 values: calculate remainder and update new dividend
//! @n: uint64_t dividend (will be updated)
//! @base: uint32_t divisor
//!
//! Summary:
//! ``uint32_t remainder = n % base;``
//! ``n = n / base;``
//!
//! Return: (uint32_t)remainder
//!
//! NOTE: macro parameter @n is evaluated multiple times,
//! beware of side effects!
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! for_each_if - helper for handling conditionals in various for_each macros
//! @condition: The condition to check
//!
//! Typical use::
//!
//! 	#define for_each_foo_bar(x, y) \'
//! 		list_for_each_entry(x, y->list, head) \'
//! 			for_each_if(x->something == SOMETHING)
//!
//! The for_each_if() macro makes the use of for_each_foo_bar() less error
//! prone.
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
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
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
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
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
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
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//! Keep unconverted code working
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
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
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Legacy last prefixes
//! REX2 M0
//! Kprobe status flags
//! When adding a new stereo mode don't forget to adjust DRM_MODE_FLAGS_3D_MAX
//! (define not exposed to user space).
//! drm_atomic_crtc_state_for_each_plane - iterate over attached planes in new state
//! @plane: the loop cursor
//! @crtc_state: the incoming CRTC state
//!
//! Similar to drm_crtc_for_each_plane(), but iterates the planes that will be
//! attached if the specified state is applied.  Useful during for example
//! in code called from &drm_mode_config_funcs.atomic_check operations, to
//! validate the incoming state.
//! DRM_FORMAT_MAX_PLANES - maximum number of planes a DRM format can have
//! See the comment in asm-generic/io.h about ioremap_np().
//! These are specified by iBCS2
//! Epoll event masks
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
// __must_not_hold() - function attribute, caller must not hold context lock
//
// Function attribute declaring that the caller must not hold the given context
// lock instance(s).
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
// Make the optimizer believe the variable can be manipulated arbitrarily.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// No space left on device
// Math result not representable
// Operation not supported on transport endpoint
// Memory page has hardware error
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
// Overflow Flag
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
// Generic interface.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// Operation is not supported
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
// in_range - Determine if a value lies within a range.
// @val: Value to test.
// @start: First value in range.
// @len: Number of values in range.
//
// This is more efficient than "if (start <= val && val < (start + len))".
// It also gives a different answer if @start + @len overflows the size of
// the type by a sufficient amount to encompass @val.  Decide for yourself
// which behaviour you want, or prove that start + len never overflow.
// Do not blindly replace one form with the other.
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
// Same as above but for u64 dividends. divisor must be a 32-bit
// number.
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
// for_each_cpu_wrap - iterate over every cpu in a mask, starting at a specified location
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
// @start: the start location
//
// The implementation does not assume any bit in @mask is set (including @start).
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
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
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
// Same as rb_first(), but O(1)
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
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
// Generate SD flag bits
// Resource extended types
// No address assigned yet
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// PT_TRACE_* event enable flags
// 0x66
//
// LPFX1
// 0x67
// x86-64 REX prefix
// 0x4X
// AVX VEX prefixes
// 2-bytes VEX prefix
// 3-bytes VEX prefix
// EVEX prefix
// x86-64 REX2 prefix
// 0xD5
// AMD XOP prefix
// 0x8F
// Escape opcodes
// Group opcodes (1-16)
// Immediates
// VEX bit flags
// VEX3 Byte2
// VEX bit fields
// EVEX Byte1
// VEX3 Byte1
// VEX2.M always 1
// VEX3 Byte2, VEX2 Byte1
// XOP Byte2
// XOP Byte3
// breakpoint has already gone
// probe is temporarily disabled
// probe is really optimized.
// NOTE:
// this flag is only for optimized_kprobe.
// probe is using ftrace
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
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
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

macro_rules! ___wait_cond_timeout {
($condition:expr, $__ret:expr) => {
{
let mut __v: bool;
let __v2: bool = $condition;
if __v2 && ($__ret) == 0 {
$__ret = 1;
}
__v = __v2 || ($__ret) == 0;
__v
}
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
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
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

/// Similar to statically_true() but produces a constant expression
///
/// To be used in conjunction with macros, such as BUILD_BUG_ON_ZERO(),
/// which require their input to be a constant expression and for which
/// statically_true() would otherwise fail.
///
/// This is a trade-off: const_true() requires all its operands to be
/// compile time constants. Else, it would always returns false even on
/// the most trivial cases like:
///
///   true || non_const_var
///
/// On the opposite, statically_true() is able to fold more complex
/// tautologies and will return true on expressions such as:
///
///   !(non_const_var * 8 % 4)
///
/// For the general case, statically_true() is better.
macro_rules! const_true {
($x:expr) => {
$x
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__10 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of {
($ptr:expr) => {
{
let mut __v: *mut drm_crtc;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut drm_crtc;
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
macro_rules! container_of__47 {
($ptr:expr) => {
{
let mut __v: *mut drm_crtc_commit;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__59!() as isize)) }) as *mut ()) as *mut drm_crtc_commit;
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
macro_rules! container_of__50 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__85!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__57 {
($ptr:expr) => {
{
let mut __v: *mut drm_crtc_commit;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__64!() as isize)) }) as *mut ()) as *mut drm_crtc_commit;
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
macro_rules! container_of__3 {
($ptr:expr) => {
{
let mut __v: *mut drm_plane;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!() as isize)) }) as *mut ()) as *mut drm_plane;
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
macro_rules! container_of__7 {
($ptr:expr) => {
{
let mut __v: *mut drm_encoder;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__9!() as isize)) }) as *mut ()) as *mut drm_encoder;
__v
}
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

macro_rules! kcsan_release {
() => {
while false {
}
};
}

macro_rules! kcsan_rmb {
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
macro_rules! list_entry__2 {
($ptr:expr) => {
container_of__3!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__3 {
($ptr:expr) => {
container_of__7!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__36 {
($ptr:expr) => {
container_of__47!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__9 {
($ptr:expr, $type:ty) => {
container_of__10!($ptr, $type)
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
macro_rules! list_first_entry__23 {
($ptr:expr) => {
list_entry__36!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__2!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__2 {
($ptr:expr) => {
list_entry__3!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__2 {
($pos:expr, $member:ident) => {
list_entry__3!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__30 {
($pos:expr, $member:ident) => {
list_entry__36!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__4 {
($pos:expr, $member:ident) => {
list_entry__2!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
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

macro_rules! offsetof__11 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, chain_node) as u64
};
}

macro_rules! offsetof__59 {
() => {
core::mem::offset_of!(drm_crtc_commit, commit_entry) as u64
};
}

macro_rules! offsetof__64 {
() => {
core::mem::offset_of!(drm_crtc_commit, flip_done) as u64
};
}

macro_rules! offsetof__9 {
() => {
core::mem::offset_of!(drm_encoder, head) as u64
};
}

macro_rules! offsetof__3 {
() => {
core::mem::offset_of!(drm_plane, head) as u64
};
}

macro_rules! offsetof__2 {
() => {
core::mem::offset_of!(drm_crtc, head) as u64
};
}

macro_rules! offsetof__85 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, commit_work) as u64
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
$flags = unsafe { _raw_spin_lock_irqsave($lock) };
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

macro_rules! ww_mutex_base_is_locked {
($b:expr) => {
unsafe { mutex_is_locked($b) }
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut system_dfl_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __bad_size_call_parameter();
    fn __xadd_wrong_size();
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_57();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn schedule_timeout(arg0: i64) -> i64;
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get() -> i64;
    fn dma_fence_release(arg0: *mut kref);
    fn dma_fence_wait_timeout(arg0: *mut dma_fence, arg1: bool, arg2: i64) -> i64;
    fn dma_fence_set_deadline(arg0: *mut dma_fence, arg1: i64);
    fn mutex_is_locked(arg0: *mut mutex) -> bool;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn wait_for_completion_interruptible(arg0: *mut completion) -> i32;
    fn wait_for_completion_timeout(arg0: *mut completion, arg1: u64) -> u64;
    fn wait_for_completion_interruptible_timeout(arg0: *mut completion, arg1: u64) -> i64;
    fn try_wait_for_completion(arg0: *mut completion) -> bool;
    fn complete_all(arg0: *mut completion);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn drm_modeset_acquire_init(arg0: *mut drm_modeset_acquire_ctx, arg1: u32);
    fn drm_modeset_acquire_fini(arg0: *mut drm_modeset_acquire_ctx);
    fn drm_modeset_drop_locks(arg0: *mut drm_modeset_acquire_ctx);
    fn drm_modeset_backoff(arg0: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn drm_modeset_lock_all_ctx(arg0: *mut drm_device, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_414();
    fn __compiletime_assert_435();
    fn __compiletime_assert_441();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __compiletime_assert_552();
    fn drm_connector_list_iter_begin(arg0: *mut drm_device, arg1: *mut drm_connector_list_iter);
    fn drm_connector_list_iter_next(arg0: *mut drm_connector_list_iter) -> *mut drm_connector;
    fn drm_connector_list_iter_end(arg0: *mut drm_connector_list_iter);
    fn drm_mode_get_hv_timing(arg0: *const drm_display_mode, arg1: *mut i32, arg2: *mut i32);
    fn drm_mode_copy(arg0: *mut drm_display_mode, arg1: *const drm_display_mode);
    fn drm_mode_equal(arg0: *const drm_display_mode, arg1: *const drm_display_mode) -> bool;
    fn drm_mode_config_reset(arg0: *mut drm_device);
    fn drm_rect_clip_scaled(arg0: *mut drm_rect, arg1: *mut drm_rect, arg2: *const drm_rect) -> bool;
    fn drm_rect_calc_hscale(arg0: *const drm_rect, arg1: *const drm_rect, arg2: i32, arg3: i32) -> i32;
    fn drm_rect_calc_vscale(arg0: *const drm_rect, arg1: *const drm_rect, arg2: i32, arg3: i32) -> i32;
    fn drm_rect_debug_print(arg0: *const u8, arg1: *const drm_rect, arg2: bool);
    fn drm_rect_rotate(arg0: *mut drm_rect, arg1: i32, arg2: i32, arg3: u32);
    fn drm_rect_rotate_inv(arg0: *mut drm_rect, arg1: i32, arg2: i32, arg3: u32);
    fn __drm_crtc_commit_free(arg0: *mut kref);
    fn drm_crtc_commit_wait(arg0: *mut drm_crtc_commit) -> i32;
    fn drm_atomic_commit_alloc(arg0: *mut drm_device) -> *mut drm_atomic_commit;
    fn __drm_atomic_commit_free(arg0: *mut kref);
    fn drm_atomic_get_crtc_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_crtc) -> *mut drm_crtc_state;
    fn drm_atomic_get_plane_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_plane) -> *mut drm_plane_state;
    fn drm_atomic_get_connector_state(arg0: *mut drm_atomic_commit, arg1: *mut drm_connector) -> *mut drm_connector_state;
    fn drm_atomic_add_encoder_bridges(arg0: *mut drm_atomic_commit, arg1: *mut drm_encoder) -> i32;
    fn drm_atomic_add_affected_connectors(arg0: *mut drm_atomic_commit, arg1: *mut drm_crtc) -> i32;
    fn drm_atomic_add_affected_planes(arg0: *mut drm_atomic_commit, arg1: *mut drm_crtc) -> i32;
    fn drm_atomic_add_affected_colorops(arg0: *mut drm_atomic_commit, arg1: *mut drm_plane) -> i32;
    fn drm_atomic_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn drm_atomic_nonblocking_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn drm_atomic_set_mode_prop_for_crtc(arg0: *mut drm_crtc_state, arg1: *mut drm_property_blob) -> i32;
    fn drm_atomic_set_crtc_for_plane(arg0: *mut drm_plane_state, arg1: *mut drm_crtc) -> i32;
    fn drm_atomic_set_fb_for_plane(arg0: *mut drm_plane_state, arg1: *mut drm_framebuffer);
    fn drm_atomic_set_crtc_for_connector(arg0: *mut drm_connector_state, arg1: *mut drm_crtc) -> i32;
    fn drm_atomic_normalize_zpos(arg0: *mut drm_device, arg1: *mut drm_atomic_commit) -> i32;
    fn drm_bridge_get(arg0: *mut drm_bridge) -> *mut drm_bridge;
    fn drm_bridge_put(arg0: *mut drm_bridge);
    fn __compiletime_assert_557();
    fn drm_bridge_chain_mode_valid(arg0: *mut drm_bridge, arg1: *const drm_display_info, arg2: *const drm_display_mode) -> i32;
    fn drm_bridge_chain_mode_set(arg0: *mut drm_bridge, arg1: *const drm_display_mode, arg2: *const drm_display_mode);
    fn drm_atomic_bridge_chain_check(arg0: *mut drm_bridge, arg1: *mut drm_crtc_state, arg2: *mut drm_connector_state) -> i32;
    fn drm_atomic_bridge_chain_disable(arg0: *mut drm_bridge, arg1: *mut drm_atomic_commit);
    fn drm_atomic_bridge_chain_post_disable(arg0: *mut drm_bridge, arg1: *mut drm_atomic_commit);
    fn drm_atomic_bridge_chain_pre_enable(arg0: *mut drm_bridge, arg1: *mut drm_atomic_commit);
    fn drm_atomic_bridge_chain_enable(arg0: *mut drm_bridge, arg1: *mut drm_atomic_commit);
    fn drm_atomic_helper_check_plane_damage(arg0: *mut drm_atomic_commit, arg1: *mut drm_plane_state);
    fn drm_format_info(arg0: u32) -> *const drm_format_info;
    fn drm_gem_plane_helper_prepare_fb(arg0: *mut drm_plane, arg1: *mut drm_plane_state) -> i32;
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn drm_self_refresh_helper_alter_state(arg0: *mut drm_atomic_commit);
    fn drm_self_refresh_helper_update_avg_times(arg0: *mut drm_atomic_commit, arg1: u32, arg2: u32);
    fn drm_dev_has_vblank(arg0: *const drm_device) -> bool;
    fn drm_crtc_vblank_count(arg0: *mut drm_crtc) -> u64;
    fn drm_crtc_next_vblank_start(arg0: *mut drm_crtc, arg1: *mut i64) -> i32;
    fn drm_crtc_send_vblank_event(arg0: *mut drm_crtc, arg1: *mut drm_pending_vblank_event);
    fn drm_crtc_vblank_get(arg0: *mut drm_crtc) -> i32;
    fn drm_crtc_vblank_put(arg0: *mut drm_crtc);
    fn drm_calc_timestamping_constants(arg0: *mut drm_crtc, arg1: *const drm_display_mode);
    fn drm_crtc_vblank_waitqueue(arg0: *mut drm_crtc) -> *mut wait_queue_head;
    fn drm_writeback_prepare_job(arg0: *mut drm_writeback_job) -> i32;
    fn drm_crtc_mode_valid(arg0: *mut drm_crtc, arg1: *const drm_display_mode) -> i32;
    fn drm_encoder_mode_valid(arg0: *mut drm_encoder, arg1: *const drm_display_mode) -> i32;
    fn drm_connector_get_single_encoder(arg0: *mut drm_connector) -> *mut drm_encoder;
    fn drm_mode_create_dumb(arg0: *mut drm_device, arg1: *mut drm_mode_create_dumb, arg2: *mut drm_file) -> i32;
    fn __drm_atomic_helper_disable_plane(arg0: *mut drm_plane, arg1: *mut drm_plane_state) -> i32;
    fn __drm_atomic_helper_set_config(arg0: *mut drm_mode_set, arg1: *mut drm_atomic_commit) -> i32;
    fn __compiletime_assert_688();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
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
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
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


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
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


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

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


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_0() };
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
unsafe { __compiletime_assert_1() };
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
unsafe { __compiletime_assert_2() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
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
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_first_entry_or_null - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note that if the list is empty, it returns NULL.
// list_last_entry_or_null - get the last element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note that if the list is empty, it returns NULL.
// list_prev_entry - get the prev element in list
// @pos:	the type * to cursor
// @member:	the name of the list_head within the struct.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
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
struct static_key_true {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits100U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits101U0,
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
struct io_bitmap {
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
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
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
core::arch::asm!("40: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 40b - ., 3; .popsection\n# {0}", const 40 as i32, options(nostack, att_syntax));
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
core::arch::asm!("41: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 41b - ., 4; .popsection\n# {0}", const 41 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
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
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[inline(always)]
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
}

#[inline(always)]
fn __arch_hweight32(mut w: u32) -> u32 {
let mut res: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight32\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntl %edi, %eax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u32, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
res
}

fn __arch_hweight8(w: u32) -> u32 {
__arch_hweight32(w & 255)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_44: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits120U0,
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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


// explicit shadow
// wait_event_timeout - sleep until a condition gets true or a timeout elapses
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
// @timeout: timeout, in jiffies
//
// The process is put to sleep (TASK_UNINTERRUPTIBLE) until the
// @condition evaluates to true. The @condition is checked each time
// the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// Returns:
// 0 if the @condition evaluated to %false after the @timeout elapsed,
// 1 if the @condition evaluated to %true after the @timeout elapsed,
// or the remaining jiffies (at least 1) if the @condition evaluated
// to %true before the @timeout elapsed.
#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


fn refcount_set(r: *mut refcount_struct, n: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
}

fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely__2!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely__2!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
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


fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
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


fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
}

fn __refcount_sub_and_test(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let old: i32 = atomic_fetch_sub_release(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if old > 0 && old == i {
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
return true;
}
if unlikely__2!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
}

fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


fn kref_init(kref: *mut kref) {
refcount_set(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }, 1i32);
}

fn kref_get(kref: *mut kref) {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*kref).refcount) });
}

fn kref_put(kref: *mut kref, release: Option<unsafe extern "C" fn(*mut kref)>) -> i32 {
if refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }) {
unsafe { release.unwrap()(kref) };
return 1;
}
0
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
struct workqueue_struct {
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
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
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_dma_fence_h_71 {
    extern_lock: *mut spinlock,
    inline_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dma_fence_h_92 {
    cb_list: list_head,
    timestamp: i64,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits69U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits69U1,
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
    __slate_bits_2: __slate_bits::__SlateBits69U2,
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
struct xol_area {
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
// rcu_pointer_handoff() - Hand off a pointer from RCU to other mechanism
// @p: The pointer to hand off
//
// This is simply an identity function, but it documents where a pointer
// is handed off from RCU to some other synchronization mechanism, for
// example, reference counting or locking.  In C11, it would map to
// kill_dependency().  It could be used as follows::
//
// 	rcu_read_lock();
// 	p = rcu_dereference(gp);
// 	long_lived = is_long_lived(p);
// 	if (long_lived) {
// 		if (!atomic_inc_not_zero(p->refcnt))
// 			long_lived = false;
// 		else
// 			p = rcu_pointer_handoff(p);
// 	}
// 	rcu_read_unlock();
#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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


fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
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


#[inline(always)]
fn msecs_to_jiffies(m: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(m) } {
if (m as i32) < 0 {
return 4611686018427387902u64;
}
return _msecs_to_jiffies(m);
} else {
return unsafe { __msecs_to_jiffies(m) };
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

fn ktime_compare(cmp1: i64, cmp2: i64) -> i32 {
if cmp1 < cmp2 {
return 1i32.wrapping_neg();
}
if cmp1 > cmp2 {
return 1;
}
0
}

fn ktime_before(cmp1: i64, cmp2: i64) -> bool {
ktime_compare(cmp1, cmp2) < 0
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


fn ktime_divns(kt: i64, div: i64) -> i64 {
// 32-bit implementation cannot handle negative divisors,
// so catch them on 64bit as well.
let mut __v0: i64;
let __ret_warn_on: i32 = (div < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("117: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 117b - ., 3; .popsection\n# {0}", const 117 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_divns_1, sym __slate_asm_string_ktime_divns_2, const 152 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("118: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 118b - ., 4; .popsection\n# {0}", const 118 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
kt / div
}

fn ktime_to_ms(kt: i64) -> i64 {
ktime_divns(kt, 1000000i64)
}

fn ktime_ms_delta(later: i64, earlier: i64) -> i64 {
ktime_to_ms(ktime_sub!(later, earlier))
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
struct dma_fence {
    __slate_anon_0: __SlateRecord_dma_fence_h_71,
    ops: *const dma_fence_ops,
    __slate_anon_2: __SlateRecord_dma_fence_h_92,
    context: u64,
    seqno: u64,
    flags: u64,
    refcount: kref,
    error: i32,
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
struct dma_fence_ops {
    get_driver_name: Option<unsafe extern "C" fn(*mut dma_fence) -> *const u8>,
    get_timeline_name: Option<unsafe extern "C" fn(*mut dma_fence) -> *const u8>,
    enable_signaling: Option<unsafe extern "C" fn(*mut dma_fence) -> bool>,
    signaled: Option<unsafe extern "C" fn(*mut dma_fence) -> bool>,
    wait: Option<unsafe extern "C" fn(*mut dma_fence, bool, i64) -> i64>,
    release: Option<unsafe extern "C" fn(*mut dma_fence)>,
    set_deadline: Option<unsafe extern "C" fn(*mut dma_fence, i64)>,
}


fn dma_fence_put(fence: *mut dma_fence) {
if !fence.is_null() {
kref_put(unsafe { core::ptr::addr_of_mut!((*fence).refcount) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut kref)>>(dma_fence_release as *const ()) });
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


fn dma_fence_wait(fence: *mut dma_fence, intr: bool) -> i64 {
let mut ret: i64 = 0;
// Since dma_fence_wait_timeout cannot timeout with
// MAX_SCHEDULE_TIMEOUT, only valid return values are
// -ERESTARTSYS and MAX_SCHEDULE_TIMEOUT.
ret = unsafe { dma_fence_wait_timeout(fence, intr, 9223372036854775807i64) };
if ret < 0 { ret } else { 0 }
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


fn init_completion(x: *mut completion) {
unsafe {
(*x).done = 0;
}
loop {
unsafe { __init_swait_queue_head(core::ptr::addr_of_mut!((*x).wait), c"&x->wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__87)) };
if true {
break;
}
}
}

static mut __key__87: lock_class_key = unsafe { core::mem::zeroed() };

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
struct drm_master {
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
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
struct user_namespace {
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


// Suits me
// Don't call further
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
struct fs_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
    __slate_bits_0: __slate_bits::__SlateBits412U0,
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
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_acquire_ctx {
    task: *mut task_struct,
    stamp: u64,
    acquired: u32,
    wounded: u16,
    is_wait_die: u16,
}


fn ww_mutex_is_locked(lock: *mut ww_mutex) -> bool {
ww_mutex_base_is_locked!(unsafe { core::ptr::addr_of_mut!((*lock).base) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_acquire_ctx {
    ww_ctx: ww_acquire_ctx,
    contended: *mut drm_modeset_lock,
    stack_depot: u32,
    locked: list_head,
    trylock_only: bool,
    interruptible: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_lock {
    mutex: ww_mutex,
    head: list_head,
}


fn drm_modeset_is_locked(lock: *mut drm_modeset_lock) -> bool {
ww_mutex_is_locked(unsafe { core::ptr::addr_of_mut!((*lock).mutex) })
}

// DRM_MODESET_LOCK_ALL_BEGIN - Helper to acquire modeset locks
// @dev: drm device
// @ctx: local modeset acquire context, will be dereferenced
// @flags: DRM_MODESET_ACQUIRE_* flags to pass to drm_modeset_acquire_init()
// @ret: local ret/err/etc variable to track error status
//
// Use these macros to simplify grabbing all modeset locks using a local
// context. This has the advantage of reducing boilerplate, but also properly
// checking return values where appropriate.
//
// Any code run between BEGIN and END will be holding the modeset locks.
//
// This must be paired with DRM_MODESET_LOCK_ALL_END(). We will jump back and
// forth between the labels on deadlock and error conditions.
//
// Drivers can acquire additional modeset locks. If any lock acquisition
// fails, the control flow needs to jump to DRM_MODESET_LOCK_ALL_END() with
// the @ret parameter containing the return value of drm_modeset_lock().
//
// Returns:
// The only possible value of ret immediately after DRM_MODESET_LOCK_ALL_BEGIN()
// is 0, so no error checking is necessary
// DRM_MODESET_LOCK_ALL_END - Helper to release and cleanup modeset locks
// @dev: drm device
// @ctx: local modeset acquire context, will be dereferenced
// @ret: local ret/err/etc variable to track error status
//
// The other side of DRM_MODESET_LOCK_ALL_BEGIN(). It will bounce back to BEGIN
// if ret is -EDEADLK.
//
// It's important that you use the same ret variable for begin and end so
// deadlock conditions are properly handled.
//
// Returns:
// ret will be untouched unless it is -EDEADLK on entry. That means that if you
// successfully acquire the locks, ret will be whatever your code sets it to. If
// there is a deadlock or other failure with acquire or backoff, ret will be set
// to that failure. In both of these cases the code between BEGIN/END will not
// be run, so the failure will reflect the inability to grab the locks.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_object {
    id: u32,
    r#type: u32,
    properties: *mut drm_object_properties,
    refcount: kref,
    free_cb: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_object_properties {
    count: i32,
    properties: [*mut drm_property; 64],
    values: [u64; 64],
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
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
#[repr(C)]
#[derive(Clone, Copy)]
struct ida {
    xa: xarray,
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
union __SlateRecord_sysctl_h_254 {
    __slate_anon_0: __SlateRecord_sysctl_h_255,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sysctl_h_255 {
    ctl_table: *const ctl_table,
    ctl_table_size: i32,
    used: i32,
    count: i32,
    nreg: i32,
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
    __slate_bits_0: __slate_bits::__SlateBits474U0,
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
struct folio_queue {
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
    __slate_bits_0: __slate_bits::__SlateBits476U0,
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
struct kmem_cache {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits501U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits501U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits501U2,
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
    __slate_bits_3: __slate_bits::__SlateBits501U3,
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct ctl_table {
    procname: *const u8,
    data: *mut (),
    maxlen: i32,
    mode: u16,
    proc_handler: Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>,
    poll: *mut ctl_table_poll,
    extra1: *mut (),
    extra2: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_header {
    __slate_anon_0: __SlateRecord_sysctl_h_254,
    unregistering: *mut completion,
    ctl_table_arg: *const ctl_table,
    root: *mut ctl_table_root,
    set: *mut ctl_table_set,
    parent: *mut ctl_dir,
    node: *mut ctl_node,
    inodes: hlist_head,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
    __slate_bits_0: __slate_bits::__SlateBits289U0,
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
unsafe { __compiletime_assert_414() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_108 {
    show: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *mut u8) -> i64>,
    show_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *mut u8) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_114 {
    store: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *const u8, u64) -> i64>,
    store_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *const u8, u64) -> i64>,
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
struct __SlateRecord_hdmi_h_214 {
    x: u16,
    y: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_217 {
    x: u16,
    y: u16,
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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_435() };
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
    __slate_bits_0: __slate_bits::__SlateBits640U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_419 {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


// no-op
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
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
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
unsafe { __compiletime_assert_441() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("442: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 442b - ., 3; .popsection\n# {0}", const 442 as i32, options(nostack, att_syntax));
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kmalloc_array_noprof(n: u64, size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely__2!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
_kmalloc_noprof(bytes, flags, token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_privacy_screen {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_daifmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_params {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter {
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
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
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
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits700U0,
    key: __SlateRecord_dynamic_debug_h_55,
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
struct platform_device {
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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
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
    __slate_bits_0: __slate_bits::__SlateBits498U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_any_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vma_offset_manager {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vram_mm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_device_h_113 {
    resources: list_head,
    final_kfree: *mut (),
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_avi_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    itc: bool,
    pixel_repeat: u8,
    colorspace: u32,
    scan_mode: u32,
    colorimetry: u32,
    picture_aspect: u32,
    active_aspect: u32,
    extended_colorimetry: u32,
    quantization_range: u32,
    nups: u32,
    video_code: u8,
    ycc_quantization_range: u32,
    content_type: u32,
    top_bar: u16,
    bottom_bar: u16,
    left_bar: u16,
    right_bar: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_drm_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    eotf: u32,
    metadata_type: u32,
    display_primaries: [__SlateRecord_hdmi_h_214; 3],
    white_point: __SlateRecord_hdmi_h_217,
    max_display_mastering_luminance: u16,
    min_display_mastering_luminance: u16,
    max_cll: u16,
    max_fall: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_spd_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    vendor: [u8; 8],
    product: [u8; 16],
    sdi: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_audio_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    channels: u8,
    coding_type: u32,
    sample_size: u32,
    sample_frequency: u32,
    coding_type_ext: u32,
    channel_allocation: u8,
    level_shift_value: u8,
    downmix_inhibit: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_vendor_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
    vic: u8,
    s3d_struct: i32,
    s3d_ext_data: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_static_metadata {
    eotf: u8,
    metadata_type: u8,
    max_cll: u16,
    max_fall: u16,
    min_cll: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_sink_metadata {
    metadata_type: u32,
    __slate_anon_1: __SlateRecord_hdmi_h_406,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_vendor_any_infoframe {
    any: __SlateRecord_hdmi_h_419,
    hdmi: hdmi_vendor_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_infoframe {
    any: hdmi_any_infoframe,
    avi: hdmi_avi_infoframe,
    spd: hdmi_spd_infoframe,
    vendor: hdmi_vendor_any_infoframe,
    audio: hdmi_audio_infoframe,
    drm: hdmi_drm_infoframe,
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
struct drm_self_refresh_data {
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
struct nsproxy {
    count: refcount_struct,
    uts_ns: *mut uts_namespace,
    ipc_ns: *mut ipc_namespace,
    mnt_ns: *mut mnt_namespace,
    pid_ns_for_children: *mut pid_namespace,
    net_ns: *mut net,
    time_ns: *mut time_namespace,
    time_ns_for_children: *mut time_namespace,
    cgroup_ns: *mut cgroup_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_node {
    ns_node: rb_node,
    ns_list_entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree {
    ns_id: u64,
    __ns_ref_active: __SlateRecord_types_h_190,
    ns_unified_node: ns_tree_node,
    ns_tree_node: ns_tree_node,
    ns_owner_node: ns_tree_node,
    ns_owner_root: ns_tree_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
    __slate_anon_0: __SlateRecord_ns_common_types_h_112,
    ns_type: u32,
    stashed: *mut dentry,
    ops: *const proc_ns_operations,
    inum: u32,
    __slate_anon_5: ns_tree,
    ns_rcu: callback_head,
}


// MFD_EXEC implied if unset
#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
    idr: idr,
    rcu: callback_head,
    pid_allocated: u32,
    memfd_noexec_scope: i32,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    child_reaper: *mut task_struct,
    pid_cachep: *mut kmem_cache,
    level: u32,
    pid_max: i32,
    parent: *mut pid_namespace,
    bacct: *mut fs_pin,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    reboot: i32,
    ns: ns_common,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_clip_rect {
    x1: u16,
    y1: u16,
    x2: u16,
    y2: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event {
    r#type: u32,
    length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event_vblank {
    base: drm_event,
    user_data: u64,
    tv_sec: u32,
    tv_usec: u32,
    sequence: u32,
    crtc_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_event_crtc_sequence {
    base: drm_event,
    user_data: u64,
    time_ns: i64,
    sequence: u64,
}


// DPMS flags
//
// bit compatible with the xorg definitions.
// Link Status options
// extended-types: rather than continue to consume a bit per type,
// grab a chunk of the bits to use as integer type id.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_fb_cmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}


// DRM_MODE_PAGE_FLIP_ASYNC
//
// Request that the page-flip is performed as soon as possible, ie. with no
// delay due to waiting for vblank. This may cause tearing to be visible on
// the screen.
//
// When used with atomic uAPI, the driver will return an error if the hardware
// doesn't support performing an asynchronous page-flip for this update.
// User-space should handle this, e.g. by falling back to a regular page-flip.
//
// Note, some hardware might need to perform one last synchronous page-flip
// before being able to switch to asynchronous page-flips. As an exception,
// the driver will return success even though that first page-flip is not
// asynchronous.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_create_dumb {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32,
    handle: u32,
    pitch: u32,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property {
    head: list_head,
    base: drm_mode_object,
    flags: u32,
    name: [u8; 32],
    num_values: u32,
    values: *mut u64,
    dev: *mut drm_device,
    enum_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property_blob {
    base: drm_mode_object,
    dev: *mut drm_device,
    head_global: list_head,
    head_file: list_head,
    length: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scrambling {
    supported: bool,
    low_rates: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scdc {
    supported: bool,
    read_request: bool,
    scrambling: drm_scrambling,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cec_msg {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_dsc_cap {
    v_1p2: bool,
    native_420: bool,
    all_bpp: bool,
    bpc_supported: u8,
    max_slices: u8,
    clk_per_slice: i32,
    max_lanes: u8,
    max_frl_rate_per_lane: u8,
    total_chunk_kbytes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_info {
    scdc: drm_scdc,
    y420_vdb_modes: [u64; 4],
    y420_cmdb_modes: [u64; 4],
    y420_dc_modes: u8,
    max_frl_rate_per_lane: u8,
    max_lanes: u8,
    dsc_cap: drm_hdmi_dsc_cap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_monitor_range_info {
    min_vfreq: u16,
    max_vfreq: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper_surface_size {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_gem_object {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_buf_attachment {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_ioctl_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_fourcc_h_77 {
    cpp: [u8; 4],
    char_per_block: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_luminance_range_info {
    min_luminance: u32,
    max_luminance: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_amd_vsdb_info {
    version: u8,
    replay_mode: bool,
    panel_type: u8,
    luminance_range1: drm_luminance_range_info,
    luminance_range2: drm_luminance_range_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_info {
    width_mm: u32,
    height_mm: u32,
    bpc: u32,
    subpixel_order: u32,
    panel_orientation: i32,
    color_formats: u32,
    bus_formats: *const u32,
    num_bus_formats: u32,
    bus_flags: u32,
    max_tmds_clock: i32,
    dvi_dual: bool,
    is_hdmi: bool,
    has_audio: bool,
    has_hdmi_infoframe: bool,
    rgb_quant_range_selectable: bool,
    edid_hdmi_rgb444_dc_modes: u8,
    edid_hdmi_ycbcr444_dc_modes: u8,
    cea_rev: u8,
    hdmi: drm_hdmi_info,
    hdr_sink_metadata: hdr_sink_metadata,
    non_desktop: bool,
    monitor_range: drm_monitor_range_info,
    luminance_range: drm_luminance_range_info,
    mso_stream_count: u8,
    mso_pixel_overlap: u8,
    max_dsc_bpp: u32,
    vics: *mut u8,
    vics_len: i32,
    quirks: u32,
    source_physical_address: u16,
    amd_vsdb: drm_amd_vsdb_info,
    panel_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_tv_margins {
    bottom: u32,
    left: u32,
    right: u32,
    top: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tv_connector_state {
    select_subconnector: u32,
    subconnector: u32,
    margins: drm_connector_tv_margins,
    legacy_mode: u32,
    mode: u32,
    brightness: u32,
    contrast: u32,
    flicker_reduction: u32,
    overscan: u32,
    saturation: u32,
    hue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iosys_map_h_110 {
    vaddr_iomem: *mut (),
    vaddr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_infoframe {
    data: hdmi_infoframe,
    set: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_state {
    broadcast_rgb: u32,
    infoframes: __SlateRecord_drm_connector_h_1088,
    is_limited_range: bool,
    output_bpc: u32,
    output_format: u32,
    tmds_char_rate: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_state {
    connector: *mut drm_connector,
    crtc: *mut drm_crtc,
    best_encoder: *mut drm_encoder,
    link_status: u32,
    state: *mut drm_atomic_commit,
    commit: *mut drm_crtc_commit,
    tv: drm_tv_connector_state,
    self_refresh_aware: bool,
    picture_aspect_ratio: u32,
    content_type: u32,
    hdcp_content_type: u32,
    scaling_mode: u32,
    content_protection: u32,
    colorspace: u32,
    color_format: u32,
    writeback_job: *mut drm_writeback_job,
    max_requested_bpc: u8,
    max_bpc: u8,
    privacy_screen_sw_state: u32,
    hdr_output_metadata: *mut drm_property_blob,
    hdmi: drm_connector_hdmi_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio_funcs {
    startup: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    prepare: Option<unsafe extern "C" fn(*mut drm_connector, *mut hdmi_codec_daifmt, *mut hdmi_codec_params) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut drm_connector)>,
    mute_stream: Option<unsafe extern "C" fn(*mut drm_connector, bool, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec_funcs {
    phys_addr_invalidate: Option<unsafe extern "C" fn(*mut drm_connector)>,
    phys_addr_set: Option<unsafe extern "C" fn(*mut drm_connector, u16)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_vblank_h_59 {
    base: drm_event,
    vbl: drm_event_vblank,
    seq: drm_event_crtc_sequence,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_infoframe_funcs {
    clear_infoframe: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    write_infoframe: Option<unsafe extern "C" fn(*mut drm_connector, *const u8, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_funcs {
    tmds_char_rate_valid: Option<unsafe extern "C" fn(*const drm_connector, *const drm_display_mode, u64) -> i32>,
    read_edid: Option<unsafe extern "C" fn(*mut drm_connector) -> *const drm_edid>,
    avi: drm_connector_infoframe_funcs,
    hdmi: drm_connector_infoframe_funcs,
    audio: drm_connector_infoframe_funcs,
    hdr_drm: drm_connector_infoframe_funcs,
    spd: drm_connector_infoframe_funcs,
}


mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits69U0 {
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
    pub struct __SlateBits69U1 {
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
    pub struct __SlateBits289U0 {
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
    pub struct __SlateBits69U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits101U0 {
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
    pub struct __SlateBits120U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits640U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits474U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits476U0 {
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
    pub struct __SlateBits100U0 {
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
    pub struct __SlateBits412U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits501U0 {
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
    pub struct __SlateBits498U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits501U1 {
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
    pub struct __SlateBits501U2 {
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
    pub struct __SlateBits501U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits843U0 {
        #[bits(1)]
        pub allow_modeset: bool,
        #[bits(1)]
        pub legacy_cursor_update: bool,
        #[bits(1)]
        pub async_update: bool,
        #[bits(1)]
        pub duplicated: bool,
        #[bits(1)]
        pub checked: bool,
        #[bits(1)]
        pub plane_color_pipeline: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits892U0 {
        #[bits(1)]
        pub planes_changed: bool,
        #[bits(1)]
        pub mode_changed: bool,
        #[bits(1)]
        pub active_changed: bool,
        #[bits(1)]
        pub connectors_changed: bool,
        #[bits(1)]
        pub zpos_changed: bool,
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits880U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits890U0 {
        #[bits(1)]
        pub hdmi_audio_spdif_playback: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits700U0 {
        #[bits(18)]
        pub lineno: u32,
        #[bits(6)]
        pub class_id: u32,
        #[bits(8)]
        pub flags: u32,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_connector, i32) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut drm_connector)>,
    detect: Option<unsafe extern "C" fn(*mut drm_connector, bool) -> u32>,
    force: Option<unsafe extern "C" fn(*mut drm_connector)>,
    fill_modes: Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_property, u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_connector)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_connector)>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_connector_state, *mut drm_property, *mut u64) -> i32>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_connector_state)>,
    oob_hotplug_event: Option<unsafe extern "C" fn(*mut drm_connector, u32)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_connector, *mut dentry)>,
    color_format: Option<unsafe extern "C" fn(*const drm_connector_state) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_cmdline_mode {
    name: [u8; 32],
    specified: bool,
    refresh_specified: bool,
    bpp_specified: bool,
    pixel_clock: u32,
    xres: i32,
    yres: i32,
    bpp: i32,
    refresh: i32,
    rb: bool,
    interlace: bool,
    cvt: bool,
    margins: bool,
    force: u32,
    rotation_reflection: u32,
    panel_orientation: i32,
    tv_margins: drm_connector_tv_margins,
    tv_mode: u32,
    tv_mode_specified: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio {
    funcs: *const drm_connector_hdmi_audio_funcs,
    codec_pdev: *mut platform_device,
    lock: mutex,
    plugged_cb: Option<unsafe extern "C" fn(*mut device, bool)>,
    plugged_cb_dev: *mut device,
    last_state: bool,
    dai_port: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi {
    vendor: [u8; 8],
    product: [u8; 16],
    supported_formats: u64,
    funcs: *const drm_connector_hdmi_funcs,
    infoframes: __SlateRecord_drm_connector_h_2062,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec {
    mutex: mutex,
    funcs: *const drm_connector_cec_funcs,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector {
    dev: *mut drm_device,
    kdev: *mut device,
    attr: *mut device_attribute,
    fwnode: *mut fwnode_handle,
    head: list_head,
    global_connector_list_entry: list_head,
    base: drm_mode_object,
    name: *mut u8,
    mutex: mutex,
    index: u32,
    connector_type: i32,
    connector_type_id: i32,
    interlace_allowed: bool,
    doublescan_allowed: bool,
    stereo_allowed: bool,
    ycbcr_420_allowed: bool,
    registration_state: u32,
    modes: list_head,
    status: u32,
    probed_modes: list_head,
    display_info: drm_display_info,
    funcs: *const drm_connector_funcs,
    edid_blob_ptr: *mut drm_property_blob,
    properties: drm_object_properties,
    scaling_mode_property: *mut drm_property,
    vrr_capable_property: *mut drm_property,
    colorspace_property: *mut drm_property,
    color_format_property: *mut drm_property,
    path_blob_ptr: *mut drm_property_blob,
    max_bpc: u32,
    max_bpc_property: *mut drm_property,
    privacy_screen: *mut drm_privacy_screen,
    privacy_screen_notifier: notifier_block,
    privacy_screen_sw_state_property: *mut drm_property,
    privacy_screen_hw_state_property: *mut drm_property,
    broadcast_rgb_property: *mut drm_property,
    polled: u8,
    dpms: i32,
    helper_private: *const drm_connector_helper_funcs,
    cmdline_mode: drm_cmdline_mode,
    force: u32,
    edid_override: *const drm_edid,
    edid_override_mutex: mutex,
    epoch_counter: u64,
    possible_encoders: u32,
    encoder: *mut drm_encoder,
    eld: [u8; 128],
    eld_mutex: mutex,
    latency_present: [bool; 2],
    video_latency: [i32; 2],
    audio_latency: [i32; 2],
    ddc: *mut i2c_adapter,
    null_edid_counter: i32,
    bad_edid_counter: u32,
    edid_corrupt: bool,
    real_edid_checksum: u8,
    debugfs_entry: *mut dentry,
    state: *mut drm_connector_state,
    tile_blob_ptr: *mut drm_property_blob,
    has_tile: bool,
    tile_group: *mut drm_tile_group,
    tile_is_single_monitor: bool,
    num_h_tile: u8,
    num_v_tile: u8,
    tile_h_loc: u8,
    tile_v_loc: u8,
    tile_h_size: u16,
    tile_v_size: u16,
    free_node: llist_node,
    hdmi: drm_connector_hdmi,
    hdmi_audio: drm_connector_hdmi_audio,
    cec: drm_connector_cec,
}


fn drm_connector_index(connector: *const drm_connector) -> u32 {
unsafe { (*connector).index }
}

fn drm_connector_is_unregistered(connector: *mut drm_connector) -> bool {
let mut __v0: u32;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_552() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*connector).registration_state) as *const u32) };
(__v0 as u32) == 2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tile_group {
    refcount: kref,
    dev: *mut drm_device,
    id: i32,
    group_data: [u8; 9],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_list_iter {
    dev: *mut drm_device,
    conn: *mut drm_connector,
}


// drm_for_each_connector_iter - connector_list iterator macro
// @connector: &struct drm_connector pointer used as cursor
// @iter: &struct drm_connector_list_iter
//
// Note that @connector is only valid within the list body, if you want to use
// @connector after calling drm_connector_list_iter_end() then you need to grab
// your own reference first using drm_connector_get().
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_mode {
    clock: i32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    flags: u32,
    crtc_clock: i32,
    crtc_hdisplay: u16,
    crtc_hblank_start: u16,
    crtc_hblank_end: u16,
    crtc_hsync_start: u16,
    crtc_hsync_end: u16,
    crtc_htotal: u16,
    crtc_hskew: u16,
    crtc_vdisplay: u16,
    crtc_vblank_start: u16,
    crtc_vblank_end: u16,
    crtc_vsync_start: u16,
    crtc_vsync_end: u16,
    crtc_vtotal: u16,
    width_mm: u16,
    height_mm: u16,
    r#type: u8,
    expose_to_userspace: bool,
    head: list_head,
    name: [u8; 32],
    status: i32,
    picture_aspect_ratio: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_funcs {
    fb_create: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, *const drm_format_info, *const drm_mode_fb_cmd2) -> *mut drm_framebuffer>,
    get_format_info: Option<unsafe extern "C" fn(u32, u64) -> *const drm_format_info>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_device, *const drm_display_mode) -> i32>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>,
    atomic_commit: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, bool) -> i32>,
    atomic_state_alloc: Option<unsafe extern "C" fn(*mut drm_device) -> *mut drm_atomic_commit>,
    atomic_state_clear: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
    atomic_state_free: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config {
    mutex: mutex,
    connection_mutex: drm_modeset_lock,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    idr_mutex: mutex,
    object_idr: idr,
    tile_idr: idr,
    fb_lock: mutex,
    num_fb: i32,
    fb_list: list_head,
    connector_list_lock: spinlock,
    num_connector: i32,
    connector_ida: ida,
    connector_list: list_head,
    connector_free_list: llist_head,
    connector_free_work: work_struct,
    num_encoder: i32,
    encoder_list: list_head,
    num_total_plane: i32,
    plane_list: list_head,
    panic_lock: raw_spinlock,
    num_colorop: i32,
    colorop_list: list_head,
    num_crtc: i32,
    crtc_list: list_head,
    property_list: list_head,
    privobj_list: list_head,
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
    funcs: *const drm_mode_config_funcs,
    poll_enabled: bool,
    poll_running: bool,
    delayed_event: bool,
    output_poll_work: delayed_work,
    blob_lock: mutex,
    property_blob_list: list_head,
    edid_property: *mut drm_property,
    dpms_property: *mut drm_property,
    path_property: *mut drm_property,
    tile_property: *mut drm_property,
    panel_type_property: *mut drm_property,
    link_status_property: *mut drm_property,
    plane_type_property: *mut drm_property,
    prop_src_x: *mut drm_property,
    prop_src_y: *mut drm_property,
    prop_src_w: *mut drm_property,
    prop_src_h: *mut drm_property,
    prop_crtc_x: *mut drm_property,
    prop_crtc_y: *mut drm_property,
    prop_crtc_w: *mut drm_property,
    prop_crtc_h: *mut drm_property,
    prop_fb_id: *mut drm_property,
    prop_in_fence_fd: *mut drm_property,
    prop_out_fence_ptr: *mut drm_property,
    prop_crtc_id: *mut drm_property,
    prop_fb_damage_clips: *mut drm_property,
    prop_active: *mut drm_property,
    prop_mode_id: *mut drm_property,
    prop_vrr_enabled: *mut drm_property,
    dvi_i_subconnector_property: *mut drm_property,
    dvi_i_select_subconnector_property: *mut drm_property,
    dp_subconnector_property: *mut drm_property,
    tv_subconnector_property: *mut drm_property,
    tv_select_subconnector_property: *mut drm_property,
    legacy_tv_mode_property: *mut drm_property,
    tv_mode_property: *mut drm_property,
    tv_left_margin_property: *mut drm_property,
    tv_right_margin_property: *mut drm_property,
    tv_top_margin_property: *mut drm_property,
    tv_bottom_margin_property: *mut drm_property,
    tv_brightness_property: *mut drm_property,
    tv_contrast_property: *mut drm_property,
    tv_flicker_reduction_property: *mut drm_property,
    tv_overscan_property: *mut drm_property,
    tv_saturation_property: *mut drm_property,
    tv_hue_property: *mut drm_property,
    scaling_mode_property: *mut drm_property,
    aspect_ratio_property: *mut drm_property,
    content_type_property: *mut drm_property,
    degamma_lut_property: *mut drm_property,
    degamma_lut_size_property: *mut drm_property,
    ctm_property: *mut drm_property,
    gamma_lut_property: *mut drm_property,
    gamma_lut_size_property: *mut drm_property,
    background_color_property: *mut drm_property,
    suggested_x_property: *mut drm_property,
    suggested_y_property: *mut drm_property,
    non_desktop_property: *mut drm_property,
    panel_orientation_property: *mut drm_property,
    writeback_fb_id_property: *mut drm_property,
    writeback_pixel_formats_property: *mut drm_property,
    writeback_out_fence_ptr_property: *mut drm_property,
    hdr_output_metadata_property: *mut drm_property,
    content_protection_property: *mut drm_property,
    hdcp_content_type_property: *mut drm_property,
    preferred_depth: u32,
    prefer_shadow: u32,
    quirk_addfb_prefer_xbgr_30bpp: bool,
    quirk_addfb_prefer_host_byte_order: bool,
    async_page_flip: bool,
    fb_modifiers_not_supported: bool,
    normalize_zpos: bool,
    modifiers_property: *mut drm_property,
    async_modifiers_property: *mut drm_property,
    size_hints_property: *mut drm_property,
    cursor_width: u32,
    cursor_height: u32,
    suspend_state: *mut drm_atomic_commit,
    helper_private: *const drm_mode_config_helper_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_device {
    if_version: i32,
    r#ref: kref,
    dev: *mut device,
    dma_dev: *mut device,
    managed: __SlateRecord_drm_device_h_113,
    driver: *const drm_driver,
    dev_private: *mut (),
    primary: *mut drm_minor,
    render: *mut drm_minor,
    accel: *mut drm_minor,
    registered: bool,
    master: *mut drm_master,
    driver_features: u32,
    unplugged: bool,
    anon_inode: *mut inode,
    unique: *mut u8,
    master_mutex: mutex,
    open_count: __SlateRecord_types_h_190,
    filelist_mutex: mutex,
    filelist: list_head,
    filelist_internal: list_head,
    clientlist_mutex: mutex,
    clientlist: list_head,
    client_sysrq_list: list_head,
    vblank_disable_immediate: bool,
    vblank: *mut drm_vblank_crtc,
    vblank_time_lock: spinlock,
    vbl_lock: spinlock,
    max_vblank_count: u32,
    vblank_event_list: list_head,
    event_lock: spinlock,
    num_crtcs: u32,
    mode_config: drm_mode_config,
    object_name_lock: mutex,
    object_name_idr: idr,
    vma_offset_manager: *mut drm_vma_offset_manager,
    vram_mm: *mut drm_vram_mm,
    switch_power_state: u32,
    fb_helper: *mut drm_fb_helper,
    debugfs_root: *mut dentry,
    gem_lru_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dump_detail {
    reason: u32,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dumper {
    list: list_head,
    dump: Option<unsafe extern "C" fn(*mut kmsg_dumper, *mut kmsg_dump_detail)>,
    max_reason: u32,
    registered: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_rect {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
}


fn drm_rect_equals(r1: *const drm_rect, r2: &drm_rect) -> bool {
(unsafe { (*r1).x1 }) == unsafe { (*(r2 as *const drm_rect)).x1 } && (unsafe { (*r1).x2 }) == unsafe { (*(r2 as *const drm_rect)).x2 } && (unsafe { (*r1).y1 }) == unsafe { (*(r2 as *const drm_rect)).y1 } && (unsafe { (*r1).y2 }) == unsafe { (*(r2 as *const drm_rect)).y2 }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_state {
    plane: *mut drm_plane,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    fence: *mut dma_fence,
    crtc_x: i32,
    crtc_y: i32,
    crtc_w: u32,
    crtc_h: u32,
    src_x: u32,
    src_y: u32,
    src_h: u32,
    src_w: u32,
    hotspot_x: i32,
    hotspot_y: i32,
    alpha: u16,
    pixel_blend_mode: u16,
    rotation: u32,
    zpos: u32,
    normalized_zpos: u32,
    color_encoding: u32,
    color_range: u32,
    fb_damage_clips: *mut drm_property_blob,
    ignore_damage_clips: bool,
    src: drm_rect,
    dst: drm_rect,
    visible: bool,
    scaling_filter: u32,
    color_pipeline: *mut drm_colorop,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
    __slate_bits_0: __slate_bits::__SlateBits880U0,
}


fn drm_plane_state_src(state: *const drm_plane_state) -> drm_rect {
let src: drm_rect = drm_rect { x1: (unsafe { (*state).src_x }) as i32, y1: (unsafe { (*state).src_y }) as i32, x2: unsafe { (*state).src_x }.wrapping_add(unsafe { (*state).src_w }) as i32, y2: unsafe { (*state).src_y }.wrapping_add(unsafe { (*state).src_h }) as i32 };
return src;
}

fn drm_plane_state_dest(state: *const drm_plane_state) -> drm_rect {
let dest: drm_rect = drm_rect { x1: unsafe { (*state).crtc_x }, y1: unsafe { (*state).crtc_y }, x2: ((unsafe { (*state).crtc_x }) as u32).wrapping_add(unsafe { (*state).crtc_w }) as i32, y2: ((unsafe { (*state).crtc_y }) as u32).wrapping_add(unsafe { (*state).crtc_h }) as i32 };
return dest;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_funcs {
    update_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_crtc, *mut drm_framebuffer, i32, i32, u32, u32, u32, u32, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    disable_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_plane)>,
    reset: Option<unsafe extern "C" fn(*mut drm_plane)>,
    set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_plane, *const drm_plane_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_plane) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_plane)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_plane_state)>,
    format_mod_supported: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
    format_mod_supported_async: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane {
    dev: *mut drm_device,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    possible_crtcs: u32,
    format_types: *mut u32,
    format_count: u32,
    format_default: bool,
    modifiers: *mut u64,
    modifier_count: u32,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    old_fb: *mut drm_framebuffer,
    funcs: *const drm_plane_funcs,
    properties: drm_object_properties,
    r#type: u32,
    index: u32,
    helper_private: *const drm_plane_helper_funcs,
    state: *mut drm_plane_state,
    alpha_property: *mut drm_property,
    zpos_property: *mut drm_property,
    rotation_property: *mut drm_property,
    blend_mode_property: *mut drm_property,
    color_encoding_property: *mut drm_property,
    color_range_property: *mut drm_property,
    color_pipeline_property: *mut drm_property,
    scaling_filter_property: *mut drm_property,
    hotspot_x_property: *mut drm_property,
    hotspot_y_property: *mut drm_property,
    kmsg_panic: kmsg_dumper,
}


fn drm_plane_index(plane: *const drm_plane) -> u32 {
unsafe { (*plane).index }
}

fn drm_plane_mask(plane: *const drm_plane) -> u32 {
(1i32 << drm_plane_index(plane)) as u32
}

// drm_for_each_plane_mask - iterate over planes specified by bitmask
// @plane: the loop cursor
// @dev: the DRM device
// @plane_mask: bitmask of plane indices
//
// Iterate over all planes specified by bitmask.
// drm_for_each_plane - iterate over all planes
// @plane: the loop cursor
// @dev: the DRM device
//
// Iterate over all planes of @dev, include primary and cursor planes.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc_entry {
    has_frame_counter: bool,
    frame: u32,
    crcs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc {
    lock: spinlock,
    source: *const u8,
    opened: bool,
    overflow: bool,
    entries: *mut drm_crtc_crc_entry,
    head: i32,
    tail: i32,
    values_cnt: u64,
    wq: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_state {
    crtc: *mut drm_crtc,
    enable: bool,
    active: bool,
    __slate_bits_0: __slate_bits::__SlateBits892U0,
    no_vblank: bool,
    plane_mask: u32,
    connector_mask: u32,
    encoder_mask: u32,
    adjusted_mode: drm_display_mode,
    mode: drm_display_mode,
    mode_blob: *mut drm_property_blob,
    degamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    background_color: u64,
    target_vblank: u32,
    async_flip: bool,
    vrr_enabled: bool,
    self_refresh_active: bool,
    scaling_filter: u32,
    sharpness_strength: u8,
    event: *mut drm_pending_vblank_event,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    cursor_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32) -> i32>,
    cursor_set2: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32, i32, i32) -> i32>,
    cursor_move: Option<unsafe extern "C" fn(*mut drm_crtc, i32, i32) -> i32>,
    gamma_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u16, *mut u16, *mut u16, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_config: Option<unsafe extern "C" fn(*mut drm_mode_set, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip_target: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_crtc_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8) -> i32>,
    verify_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8, *mut u64) -> i32>,
    get_crc_sources: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u64) -> *const *const u8>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_crtc_state)>,
    get_vblank_counter: Option<unsafe extern "C" fn(*mut drm_crtc) -> u32>,
    enable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    disable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    get_vblank_timestamp: Option<unsafe extern "C" fn(*mut drm_crtc, *mut i32, *mut i64, bool) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc {
    dev: *mut drm_device,
    port: *mut device_node,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    primary: *mut drm_plane,
    cursor: *mut drm_plane,
    index: u32,
    cursor_x: i32,
    cursor_y: i32,
    enabled: bool,
    mode: drm_display_mode,
    hwmode: drm_display_mode,
    x: i32,
    y: i32,
    funcs: *const drm_crtc_funcs,
    gamma_size: u32,
    gamma_store: *mut u16,
    helper_private: *const drm_crtc_helper_funcs,
    properties: drm_object_properties,
    scaling_filter_property: *mut drm_property,
    sharpness_strength_property: *mut drm_property,
    state: *mut drm_crtc_state,
    commit_list: list_head,
    commit_lock: spinlock,
    debugfs_entry: *mut dentry,
    crc: drm_crtc_crc,
    fence_context: u32,
    fence_lock: spinlock,
    fence_seqno: u64,
    timeline_name: [u8; 32],
    self_refresh_data: *mut drm_self_refresh_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_set {
    fb: *mut drm_framebuffer,
    crtc: *mut drm_crtc,
    mode: *mut drm_display_mode,
    x: u32,
    y: u32,
    connectors: *mut *mut drm_connector,
    num_connectors: u64,
}


fn drm_crtc_index(crtc: *const drm_crtc) -> u32 {
unsafe { (*crtc).index }
}

fn drm_crtc_mask(crtc: *const drm_crtc) -> u32 {
(1i32 << drm_crtc_index(crtc)) as u32
}

// drm_for_each_crtc - iterate over all CRTCs
// @crtc: a &struct drm_crtc as the loop cursor
// @dev: the &struct drm_device
//
// Iterate over all CRTCs of @dev.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_commit {
    crtc: *mut drm_crtc,
    r#ref: kref,
    flip_done: completion,
    hw_done: completion,
    cleanup_done: completion,
    commit_entry: list_head,
    event: *mut drm_pending_vblank_event,
    abort_completion: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_colorops_state {
    ptr: *mut drm_colorop,
    state: *mut drm_colorop_state,
    old_state: *mut drm_colorop_state,
    new_state: *mut drm_colorop_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_planes_state {
    ptr: *mut drm_plane,
    state_to_destroy: *mut drm_plane_state,
    old_state: *mut drm_plane_state,
    new_state: *mut drm_plane_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_crtcs_state {
    ptr: *mut drm_crtc,
    state_to_destroy: *mut drm_crtc_state,
    old_state: *mut drm_crtc_state,
    new_state: *mut drm_crtc_state,
    commit: *mut drm_crtc_commit,
    out_fence_ptr: *mut i32,
    last_vblank_count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_connnectors_state {
    ptr: *mut drm_connector,
    state_to_destroy: *mut drm_connector_state,
    old_state: *mut drm_connector_state,
    new_state: *mut drm_connector_state,
    out_fence_ptr: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_state_funcs {
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_private_obj) -> *mut drm_private_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_private_obj) -> *mut drm_private_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_private_obj, *mut drm_private_state)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_private_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_obj {
    dev: *mut drm_device,
    head: list_head,
    lock: drm_modeset_lock,
    state: *mut drm_private_state,
    funcs: *const drm_private_state_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_state {
    state: *mut drm_atomic_commit,
    obj: *mut drm_private_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_private_objs_state {
    ptr: *mut drm_private_obj,
    state_to_destroy: *mut drm_private_state,
    old_state: *mut drm_private_state,
    new_state: *mut drm_private_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_atomic_commit {
    r#ref: kref,
    dev: *mut drm_device,
    __slate_bits_0: __slate_bits::__SlateBits843U0,
    colorops: *mut __drm_colorops_state,
    planes: *mut __drm_planes_state,
    crtcs: *mut __drm_crtcs_state,
    num_connector: i32,
    connectors: *mut __drm_connnectors_state,
    num_private_objs: i32,
    private_objs: *mut __drm_private_objs_state,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    fake_commit: *mut drm_crtc_commit,
    commit_work: work_struct,
}


fn drm_crtc_commit_get(commit: *mut drm_crtc_commit) -> *mut drm_crtc_commit {
kref_get(unsafe { core::ptr::addr_of_mut!((*commit).r#ref) });
commit
}

fn drm_crtc_commit_put(commit: *mut drm_crtc_commit) {
kref_put(unsafe { core::ptr::addr_of_mut!((*commit).r#ref) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut kref)>>(__drm_crtc_commit_free as *const ()) });
}

fn drm_atomic_commit_get(state: *mut drm_atomic_commit) -> *mut drm_atomic_commit {
kref_get(unsafe { core::ptr::addr_of_mut!((*state).r#ref) });
state
}

fn drm_atomic_commit_put(state: *mut drm_atomic_commit) {
kref_put(unsafe { core::ptr::addr_of_mut!((*state).r#ref) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut kref)>>(__drm_atomic_commit_free as *const ()) });
}

fn drm_atomic_get_old_crtc_state(state: *const drm_atomic_commit, crtc: *mut drm_crtc) -> *mut drm_crtc_state {
unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(drm_crtc_index(crtc as *const drm_crtc) as isize)).old_state }
}

fn drm_atomic_get_new_crtc_state(state: *const drm_atomic_commit, crtc: *mut drm_crtc) -> *mut drm_crtc_state {
unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(drm_crtc_index(crtc as *const drm_crtc) as isize)).new_state }
}

fn drm_atomic_get_old_plane_state(state: *const drm_atomic_commit, plane: *mut drm_plane) -> *mut drm_plane_state {
unsafe { (*unsafe { (*state).planes }.wrapping_offset(drm_plane_index(plane as *const drm_plane) as isize)).old_state }
}

fn drm_atomic_get_new_plane_state(state: *const drm_atomic_commit, plane: *mut drm_plane) -> *mut drm_plane_state {
unsafe { (*unsafe { (*state).planes }.wrapping_offset(drm_plane_index(plane as *const drm_plane) as isize)).new_state }
}

fn drm_atomic_get_new_connector_state(state: *const drm_atomic_commit, connector: *mut drm_connector) -> *mut drm_connector_state {
let index: i32 = drm_connector_index(connector as *const drm_connector) as i32;
if index >= unsafe { (*state).num_connector } {
return core::ptr::null_mut::<drm_connector_state>();
}
unsafe { (*unsafe { (*state).connectors }.wrapping_offset(index as isize)).new_state }
}

// for_each_oldnew_connector_in_state - iterate over all connectors in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @connector: &struct drm_connector iteration cursor
// @old_connector_state: &struct drm_connector_state iteration cursor for the
// 	old state
// @new_connector_state: &struct drm_connector_state iteration cursor for the
// 	new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all connectors in an atomic update, tracking both old and
// new state. This is useful in places where the state delta needs to be
// considered, for example in atomic check functions.
// Only to avoid unused-but-set-variable warning
// for_each_old_connector_in_state - iterate over all connectors in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @connector: &struct drm_connector iteration cursor
// @old_connector_state: &struct drm_connector_state iteration cursor for the
// 	old state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all connectors in an atomic update, tracking only the old
// state. This is useful in disable functions, where we need the old state the
// hardware is still in.
// Only to avoid unused-but-set-variable warning
// for_each_new_connector_in_state - iterate over all connectors in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @connector: &struct drm_connector iteration cursor
// @new_connector_state: &struct drm_connector_state iteration cursor for the
// 	new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all connectors in an atomic update, tracking only the new
// state. This is useful in enable functions, where we need the new state the
// hardware should be in when the atomic commit operation has completed.
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
// for_each_oldnew_crtc_in_state - iterate over all CRTCs in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @crtc: &struct drm_crtc iteration cursor
// @old_crtc_state: &struct drm_crtc_state iteration cursor for the old state
// @new_crtc_state: &struct drm_crtc_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all CRTCs in an atomic update, tracking both old and
// new state. This is useful in places where the state delta needs to be
// considered, for example in atomic check functions.
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
// for_each_old_crtc_in_state - iterate over all CRTCs in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @crtc: &struct drm_crtc iteration cursor
// @old_crtc_state: &struct drm_crtc_state iteration cursor for the old state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all CRTCs in an atomic update, tracking only the old
// state. This is useful in disable functions, where we need the old state the
// hardware is still in.
// Only to avoid unused-but-set-variable warning
// for_each_new_crtc_in_state - iterate over all CRTCs in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @crtc: &struct drm_crtc iteration cursor
// @new_crtc_state: &struct drm_crtc_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all CRTCs in an atomic update, tracking only the new
// state. This is useful in enable functions, where we need the new state the
// hardware should be in when the atomic commit operation has completed.
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
// for_each_oldnew_colorop_in_state - iterate over all colorops in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @colorop: &struct drm_colorop iteration cursor
// @old_colorop_state: &struct drm_colorop_state iteration cursor for the old state
// @new_colorop_state: &struct drm_colorop_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all colorops in an atomic update, tracking both old and
// new state. This is useful in places where the state delta needs to be
// considered, for example in atomic check functions.
// Only to avoid unused-but-set-variable warning
// for_each_new_colorop_in_state - iterate over all colorops in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @colorop: &struct drm_colorop iteration cursor
// @new_colorop_state: &struct drm_colorop_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all colorops in an atomic update, tracking new state. This is
// useful in places where the state delta needs to be considered, for example in
// atomic check functions.
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
// for_each_oldnew_plane_in_state - iterate over all planes in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @plane: &struct drm_plane iteration cursor
// @old_plane_state: &struct drm_plane_state iteration cursor for the old state
// @new_plane_state: &struct drm_plane_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all planes in an atomic update, tracking both old and
// new state. This is useful in places where the state delta needs to be
// considered, for example in atomic check functions.
// Only to avoid unused-but-set-variable warning
// for_each_old_plane_in_state - iterate over all planes in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @plane: &struct drm_plane iteration cursor
// @old_plane_state: &struct drm_plane_state iteration cursor for the old state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all planes in an atomic update, tracking only the old
// state. This is useful in disable functions, where we need the old state the
// hardware is still in.
// for_each_new_plane_in_state - iterate over all planes in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @plane: &struct drm_plane iteration cursor
// @new_plane_state: &struct drm_plane_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all planes in an atomic update, tracking only the new
// state. This is useful in enable functions, where we need the new state the
// hardware should be in when the atomic commit operation has completed.
// Only to avoid unused-but-set-variable warning
// Only to avoid unused-but-set-variable warning
// for_each_oldnew_private_obj_in_state - iterate over all private objects in an atomic update
// @__state: &struct drm_atomic_commit pointer
// @obj: &struct drm_private_obj iteration cursor
// @old_obj_state: &struct drm_private_state iteration cursor for the old state
// @new_obj_state: &struct drm_private_state iteration cursor for the new state
// @__i: int iteration cursor, for macro-internal use
//
// This iterates over all private objects in an atomic update, tracking both
// old and new state. This is useful in places where the state delta needs
// to be considered, for example in atomic check functions.
fn drm_atomic_crtc_needs_modeset(state: *const drm_crtc_state) -> bool {
((unsafe { (*state).__slate_bits_0.__get_mode_changed() }) as i32) != 0 || ((unsafe { (*state).__slate_bits_0.__get_active_changed() }) as i32) != 0 || ((unsafe { (*state).__slate_bits_0.__get_connectors_changed() }) as i32) != 0
}

fn drm_atomic_crtc_effectively_active(state: *const drm_crtc_state) -> bool {
(unsafe { (*state).active }) || unsafe { (*state).self_refresh_active }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_bus_cfg {
    format: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_bridge_state {
    base: drm_private_state,
    bridge: *mut drm_bridge,
    input_bus_cfg: drm_bus_cfg,
    output_bus_cfg: drm_bus_cfg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    late_register: Option<unsafe extern "C" fn(*mut drm_encoder) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_encoder, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    name: *mut u8,
    encoder_type: i32,
    index: u32,
    possible_crtcs: u32,
    possible_clones: u32,
    crtc: *mut drm_crtc,
    bridge_chain: list_head,
    bridge_chain_mutex: mutex,
    funcs: *const drm_encoder_funcs,
    helper_private: *const drm_encoder_helper_funcs,
    debugfs_entry: *mut dentry,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_check_modeset: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_check_wb_connector_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_check_plane_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_check_crtc_primary_plane: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_check_planes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_check: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_encoder_bridge_disable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_crtc_disable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_encoder_bridge_post_disable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_update_legacy_modeset_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_calc_timestamping_constants: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_crtc_set_mode: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_modeset_disables: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_writebacks: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_encoder_bridge_pre_enable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_crtc_enable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_encoder_bridge_enable: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_modeset_enables: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_wait_for_fences: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_wait_for_vblanks: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_wait_for_flip_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_tail: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_tail_rpm: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_async_check: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_async_commit: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_setup_commit: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_wait_for_dependencies: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_fake_vblank: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_hw_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_cleanup_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_prepare_planes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_unprepare_planes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_planes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_planes_on_crtc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_disable_planes_on_crtc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_cleanup_planes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_swap_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_update_plane: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_disable_plane: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_set_config: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_disable_all: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_reset_crtc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_shutdown: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_duplicate_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_suspend: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_commit_duplicated_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_resume: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_page_flip: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_page_flip_target: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_bridge_propagate_bus_fmt: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_atomic_helper_bridge_get_hdmi_output_bus_fmts: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous", sym drm_atomic_helper_check_modeset, sym drm_atomic_helper_check_wb_connector_state, sym drm_atomic_helper_check_plane_state, sym drm_atomic_helper_check_crtc_primary_plane, sym drm_atomic_helper_check_planes, sym drm_atomic_helper_check, sym drm_atomic_helper_commit_encoder_bridge_disable, sym drm_atomic_helper_commit_crtc_disable, sym drm_atomic_helper_commit_encoder_bridge_post_disable, sym drm_atomic_helper_update_legacy_modeset_state, sym drm_atomic_helper_calc_timestamping_constants, sym drm_atomic_helper_commit_crtc_set_mode, sym drm_atomic_helper_commit_modeset_disables, sym drm_atomic_helper_commit_writebacks, sym drm_atomic_helper_commit_encoder_bridge_pre_enable, sym drm_atomic_helper_commit_crtc_enable, sym drm_atomic_helper_commit_encoder_bridge_enable, sym drm_atomic_helper_commit_modeset_enables, sym drm_atomic_helper_wait_for_fences, sym drm_atomic_helper_wait_for_vblanks, sym drm_atomic_helper_wait_for_flip_done, sym drm_atomic_helper_commit_tail, sym drm_atomic_helper_commit_tail_rpm, sym drm_atomic_helper_async_check, sym drm_atomic_helper_async_commit, sym drm_atomic_helper_commit, sym drm_atomic_helper_setup_commit, sym drm_atomic_helper_wait_for_dependencies, sym drm_atomic_helper_fake_vblank, sym drm_atomic_helper_commit_hw_done, sym drm_atomic_helper_commit_cleanup_done, sym drm_atomic_helper_prepare_planes, sym drm_atomic_helper_unprepare_planes, sym drm_atomic_helper_commit_planes, sym drm_atomic_helper_commit_planes_on_crtc, sym drm_atomic_helper_disable_planes_on_crtc, sym drm_atomic_helper_cleanup_planes, sym drm_atomic_helper_swap_state, sym drm_atomic_helper_update_plane, sym drm_atomic_helper_disable_plane, sym drm_atomic_helper_set_config, sym drm_atomic_helper_disable_all, sym drm_atomic_helper_reset_crtc, sym drm_atomic_helper_shutdown, sym drm_atomic_helper_duplicate_state, sym drm_atomic_helper_suspend, sym drm_atomic_helper_commit_duplicated_state, sym drm_atomic_helper_resume, sym drm_atomic_helper_page_flip, sym drm_atomic_helper_page_flip_target, sym drm_atomic_helper_bridge_propagate_bus_fmt, sym drm_atomic_helper_bridge_get_hdmi_output_bus_fmts, options(att_syntax));

fn drm_encoder_index(encoder: *const drm_encoder) -> u32 {
unsafe { (*encoder).index }
}

fn drm_encoder_mask(encoder: *const drm_encoder) -> u32 {
(1i32 << drm_encoder_index(encoder)) as u32
}

fn drm_encoder_crtc_ok(encoder: *mut drm_encoder, crtc: *mut drm_crtc) -> bool {
(unsafe { (*encoder).possible_crtcs }) & drm_crtc_mask(crtc as *const drm_crtc) != 0
}

// drm_for_each_encoder_mask - iterate over encoders specified by bitmask
// @encoder: the loop cursor
// @dev: the DRM device
// @encoder_mask: bitmask of encoder indices
//
// Iterate over all encoders specified by bitmask.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_helper_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_crtc, i32)>,
    prepare: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    commit: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_display_mode) -> i32>,
    mode_fixup: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_display_mode, *mut drm_display_mode) -> bool>,
    mode_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_display_mode, *mut drm_display_mode, i32, i32, *mut drm_framebuffer) -> i32>,
    mode_set_nofb: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    mode_set_base: Option<unsafe extern "C" fn(*mut drm_crtc, i32, i32, *mut drm_framebuffer) -> i32>,
    disable: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit) -> i32>,
    atomic_begin: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    atomic_flush: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_atomic_commit)>,
    get_scanout_position: Option<unsafe extern "C" fn(*mut drm_crtc, bool, *mut i32, *mut i32, *mut i64, *mut i64, *const drm_display_mode) -> bool>,
    handle_vblank_timeout: Option<unsafe extern "C" fn(*mut drm_crtc) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_helper_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_encoder, i32)>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_encoder, *const drm_display_mode) -> i32>,
    mode_fixup: Option<unsafe extern "C" fn(*mut drm_encoder, *const drm_display_mode, *mut drm_display_mode) -> bool>,
    prepare: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    commit: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    mode_set: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_display_mode, *mut drm_display_mode)>,
    atomic_mode_set: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_crtc_state, *mut drm_connector_state)>,
    detect: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_connector) -> u32>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_atomic_commit)>,
    disable: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    enable: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_encoder, *mut drm_crtc_state, *mut drm_connector_state) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_helper_funcs {
    get_modes: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    detect_ctx: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_modeset_acquire_ctx, bool) -> i32>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_display_mode) -> i32>,
    mode_valid_ctx: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_display_mode, *mut drm_modeset_acquire_ctx, *mut i32) -> i32>,
    best_encoder: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_encoder>,
    atomic_best_encoder: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit) -> *mut drm_encoder>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit) -> i32>,
    atomic_commit: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit)>,
    prepare_writeback_job: Option<unsafe extern "C" fn(*mut drm_writeback_connector, *mut drm_writeback_job) -> i32>,
    cleanup_writeback_job: Option<unsafe extern "C" fn(*mut drm_writeback_connector, *mut drm_writeback_job)>,
    enable_hpd: Option<unsafe extern "C" fn(*mut drm_connector)>,
    disable_hpd: Option<unsafe extern "C" fn(*mut drm_connector)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_helper_funcs {
    prepare_fb: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state) -> i32>,
    cleanup_fb: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    begin_fb_access: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state) -> i32>,
    end_fb_access: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit) -> i32>,
    atomic_update: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    atomic_async_check: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit, bool) -> i32>,
    atomic_async_update: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_atomic_commit)>,
    get_scanout_buffer: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_scanout_buffer) -> i32>,
    panic_flush: Option<unsafe extern "C" fn(*mut drm_plane)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_helper_funcs {
    atomic_commit_tail: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
    atomic_commit_setup: Option<unsafe extern "C" fn(*mut drm_atomic_commit) -> i32>,
}


fn drm_atomic_plane_enabling(old_plane_state: *mut drm_plane_state, new_plane_state: *mut drm_plane_state) -> bool {
// When enabling a plane, CRTC and FB should always be set together.
// Anything else should be considered a bug in the atomic core, so we
// gently warn about it.
let mut __v0: i64;
let __ret_warn_on: i32 = (unsafe { (*new_plane_state).crtc }.is_null() && !unsafe { (*new_plane_state).fb }.is_null() || !unsafe { (*new_plane_state).crtc }.is_null() && unsafe { (*new_plane_state).fb }.is_null()) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("553: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 553b - ., 3; .popsection\n# {0}", const 553 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_plane_enabling_1, sym __slate_asm_string_drm_atomic_plane_enabling_2, const 258 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("554: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 554b - ., 4; .popsection\n# {0}", const 554 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { (*old_plane_state).crtc }.is_null() && !unsafe { (*new_plane_state).crtc }.is_null()
}

fn drm_atomic_plane_disabling(old_plane_state: *mut drm_plane_state, new_plane_state: *mut drm_plane_state) -> bool {
// When disabling a plane, CRTC and FB should always be NULL together.
// Anything else should be considered a bug in the atomic core, so we
// gently warn about it.
let mut __v0: i64;
let __ret_warn_on: i32 = (unsafe { (*new_plane_state).crtc }.is_null() && !unsafe { (*new_plane_state).fb }.is_null() || !unsafe { (*new_plane_state).crtc }.is_null() && unsafe { (*new_plane_state).fb }.is_null()) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("555: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 555b - ., 3; .popsection\n# {0}", const 555 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_plane_disabling_1, sym __slate_asm_string_drm_atomic_plane_disabling_2, const 285 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("556: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 556b - ., 4; .popsection\n# {0}", const 556 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
!unsafe { (*old_plane_state).crtc }.is_null() && unsafe { (*new_plane_state).crtc }.is_null()
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_bridge_get_hdmi_output_bus_fmts")]
extern "C" fn drm_atomic_helper_bridge_get_hdmi_output_bus_fmts(bridge: *mut drm_bridge, bridge_state: *mut drm_bridge_state, crtc_state: *mut drm_crtc_state, conn_state: *mut drm_connector_state, num_output_fmts: *mut u32) -> *mut u32 {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut num_fmts: u32 = 0;
// bridge->supported_formats is a bit field of BIT(enum drm_output_color_format)
// values. The smallest hweight that is smaller than or equal to
// %DRM_OUTPUT_COLOR_FORMAT_COUNT will do for counting set bits here.
loop {
if !(!const_true!(false)) {
unsafe { __compiletime_assert_688() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
let __v3: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe((*bridge).supported_formats) } {
((((unsafe { (*bridge).supported_formats }) as u64) & 1u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 2u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 4u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 8u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 16u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 32u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 64u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 128u64 != 0) as i32) as u32
} else {
__arch_hweight8(unsafe { (*bridge).supported_formats })
};
_res = _kmalloc_array_noprof(__v3 as u64, core::mem::size_of::<u32>() as u64, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
let __v6: u32 = if unsafe { __slate_intrinsic_1c5364fcd9da51fe((*bridge).supported_formats) } {
((((unsafe { (*bridge).supported_formats }) as u64) & 1u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 2u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 4u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 8u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 16u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 32u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 64u64 != 0) as i32).wrapping_add((((unsafe { (*bridge).supported_formats }) as u64) & 128u64 != 0) as i32) as u32
} else {
__arch_hweight8(unsafe { (*bridge).supported_formats })
};
_res = _kmalloc_array_noprof(__v6 as u64, core::mem::size_of::<u32>() as u64, 3264u32, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
}
__v2 = _res;
__v1 = __v2;
let out_fmts: *mut u32 = __v1 as *mut u32;
if out_fmts.is_null() {
unsafe {
*num_output_fmts = 0;
}
return core::ptr::null_mut::<u32>();
}
match unsafe { (*conn_state).hdmi.output_bpc } {
12 => {
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(0i32) != 0 {
let __v8: u32 = num_fmts;
num_fmts = __v8.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v8 as isize) = 4121;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(1i32) != 0 {
let __v10: u32 = num_fmts;
num_fmts = __v10.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v10 as isize) = 8233;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(2i32) != 0 {
let __v12: u32 = num_fmts;
num_fmts = __v12.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v12 as isize) = 8224;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(3i32) != 0 {
let __v14: u32 = num_fmts;
num_fmts = __v14.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v14 as isize) = 8232;
}
}
}
10 => {
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(0i32) != 0 {
let __v17: u32 = num_fmts;
num_fmts = __v17.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v17 as isize) = 4120;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(1i32) != 0 {
let __v19: u32 = num_fmts;
num_fmts = __v19.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v19 as isize) = 8214;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(2i32) != 0 {
let __v21: u32 = num_fmts;
num_fmts = __v21.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v21 as isize) = 8218;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(3i32) != 0 {
let __v23: u32 = num_fmts;
num_fmts = __v23.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v23 as isize) = 8231;
}
}
}
_ => {
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(0i32) != 0 {
let __v25: u32 = num_fmts;
num_fmts = __v25.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v25 as isize) = 4106;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(1i32) != 0 {
let __v27: u32 = num_fmts;
num_fmts = __v27.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v27 as isize) = 8229;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(2i32) != 0 {
let __v29: u32 = num_fmts;
num_fmts = __v29.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v29 as isize) = 8207;
}
}
if ((unsafe { (*bridge).supported_formats }) as u64) & BIT!(3i32) != 0 {
let __v31: u32 = num_fmts;
num_fmts = __v31.wrapping_add(1);
unsafe {
*out_fmts.wrapping_offset(__v31 as isize) = 8230;
}
}
}
}
unsafe {
*num_output_fmts = num_fmts;
}
out_fmts
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_bridge_funcs {
    attach: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_encoder, u32) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_bridge)>,
    detach: Option<unsafe extern "C" fn(*mut drm_bridge)>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_bridge, *const drm_display_info, *const drm_display_mode) -> i32>,
    mode_fixup: Option<unsafe extern "C" fn(*mut drm_bridge, *const drm_display_mode, *mut drm_display_mode) -> bool>,
    mode_set: Option<unsafe extern "C" fn(*mut drm_bridge, *const drm_display_mode, *const drm_display_mode)>,
    atomic_pre_enable: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_atomic_commit)>,
    atomic_enable: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_atomic_commit)>,
    atomic_disable: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_atomic_commit)>,
    atomic_post_disable: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_atomic_commit)>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_bridge) -> *mut drm_bridge_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_bridge_state)>,
    atomic_get_output_bus_fmts: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_bridge_state, *mut drm_crtc_state, *mut drm_connector_state, *mut u32) -> *mut u32>,
    atomic_get_input_bus_fmts: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_bridge_state, *mut drm_crtc_state, *mut drm_connector_state, u32, *mut u32) -> *mut u32>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_bridge_state, *mut drm_crtc_state, *mut drm_connector_state) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_bridge) -> *mut drm_bridge_state>,
    detect: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector) -> u32>,
    get_modes: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector) -> i32>,
    edid_read: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector) -> *const drm_edid>,
    hpd_notify: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector, u32)>,
    hpd_enable: Option<unsafe extern "C" fn(*mut drm_bridge)>,
    hpd_disable: Option<unsafe extern "C" fn(*mut drm_bridge)>,
    hdmi_tmds_char_rate_valid: Option<unsafe extern "C" fn(*const drm_bridge, *const drm_display_mode, u64) -> i32>,
    hdmi_clear_avi_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge) -> i32>,
    hdmi_write_avi_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge, *const u8, u64) -> i32>,
    hdmi_clear_hdmi_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge) -> i32>,
    hdmi_write_hdmi_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge, *const u8, u64) -> i32>,
    hdmi_clear_hdr_drm_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge) -> i32>,
    hdmi_write_hdr_drm_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge, *const u8, u64) -> i32>,
    hdmi_clear_spd_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge) -> i32>,
    hdmi_write_spd_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge, *const u8, u64) -> i32>,
    hdmi_clear_audio_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge) -> i32>,
    hdmi_write_audio_infoframe: Option<unsafe extern "C" fn(*mut drm_bridge, *const u8, u64) -> i32>,
    hdmi_audio_startup: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector) -> i32>,
    hdmi_audio_prepare: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector, *mut hdmi_codec_daifmt, *mut hdmi_codec_params) -> i32>,
    hdmi_audio_shutdown: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector)>,
    hdmi_audio_mute_stream: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector, bool, i32) -> i32>,
    hdmi_cec_init: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector) -> i32>,
    hdmi_cec_enable: Option<unsafe extern "C" fn(*mut drm_bridge, bool) -> i32>,
    hdmi_cec_log_addr: Option<unsafe extern "C" fn(*mut drm_bridge, u8) -> i32>,
    hdmi_cec_transmit: Option<unsafe extern "C" fn(*mut drm_bridge, u8, u32, *mut cec_msg) -> i32>,
    dp_audio_startup: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector) -> i32>,
    dp_audio_prepare: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector, *mut hdmi_codec_daifmt, *mut hdmi_codec_params) -> i32>,
    dp_audio_shutdown: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector)>,
    dp_audio_mute_stream: Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_connector, bool, i32) -> i32>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_bridge, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_bridge_timings {
    input_bus_flags: u32,
    setup_time_ps: u32,
    hold_time_ps: u32,
    dual_link: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_bridge {
    base: drm_private_obj,
    dev: *mut drm_device,
    encoder: *mut drm_encoder,
    chain_node: list_head,
    of_node: *mut device_node,
    list: list_head,
    timings: *const drm_bridge_timings,
    funcs: *const drm_bridge_funcs,
    container: *mut (),
    refcount: kref,
    unplugged: bool,
    driver_private: *mut (),
    ops: u32,
    r#type: i32,
    interlace_allowed: bool,
    ycbcr_420_allowed: bool,
    pre_enable_prev_first: bool,
    support_hdcp: bool,
    ddc: *mut i2c_adapter,
    vendor: *const u8,
    product: *const u8,
    supported_formats: u32,
    max_bpc: u32,
    hdmi_cec_dev: *mut device,
    hdmi_audio_dev: *mut device,
    hdmi_audio_max_i2s_playback_channels: i32,
    hdmi_audio_i2s_formats: u64,
    __slate_bits_0: __slate_bits::__SlateBits890U0,
    hdmi_audio_dai_port: i32,
    hdmi_cec_adapter_name: *const u8,
    hdmi_cec_available_las: u8,
    hpd_mutex: mutex,
    hpd_state_mutex: mutex,
    hpd_cb: Option<unsafe extern "C" fn(*mut (), u32)>,
    hpd_data: *mut (),
    next_bridge: *mut drm_bridge,
}


fn drm_bridge_chain_get_first_bridge(encoder: *mut drm_encoder) -> *mut drm_bridge {
let mut __v0: *mut drm_bridge;
let head__: *mut list_head = unsafe { core::ptr::addr_of_mut!((*encoder).bridge_chain) };
let mut __v1: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_557() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v1;
let __v3: *mut drm_bridge = if pos__ != head__ {
list_entry__9!(pos__, drm_bridge)
} else {
core::ptr::null_mut::<drm_bridge>()
};
__v0 = __v3;
unsafe { drm_bridge_get(__v0) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_state {
    colorop: *mut drm_colorop,
    bypass: bool,
    curve_1d_type: u32,
    multiplier: u64,
    data: *mut drm_property_blob,
    lut1d_interpolation: u32,
    lut3d_interpolation: u32,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_funcs {
    destroy: Option<unsafe extern "C" fn(*mut drm_colorop)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop {
    dev: *mut drm_device,
    head: list_head,
    index: u32,
    base: drm_mode_object,
    plane: *mut drm_plane,
    state: *mut drm_colorop_state,
    properties: drm_object_properties,
    r#type: u32,
    next: *mut drm_colorop,
    type_property: *mut drm_property,
    bypass_property: *mut drm_property,
    size: u32,
    lut1d_interpolation_property: *mut drm_property,
    curve_1d_type_property: *mut drm_property,
    multiplier_property: *mut drm_property,
    size_property: *mut drm_property,
    lut3d_interpolation_property: *mut drm_property,
    data_property: *mut drm_property,
    next_property: *mut drm_property,
    funcs: *const drm_colorop_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
    load: Option<unsafe extern "C" fn(*mut drm_device, u64) -> i32>,
    open: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file) -> i32>,
    postclose: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file)>,
    unload: Option<unsafe extern "C" fn(*mut drm_device)>,
    release: Option<unsafe extern "C" fn(*mut drm_device)>,
    master_set: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, bool)>,
    master_drop: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_minor)>,
    gem_create_object: Option<unsafe extern "C" fn(*mut drm_device, u64) -> *mut drm_gem_object>,
    prime_handle_to_fd: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, u32, u32, *mut i32) -> i32>,
    prime_fd_to_handle: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, i32, *mut u32) -> i32>,
    gem_prime_import: Option<unsafe extern "C" fn(*mut drm_device, *mut dma_buf) -> *mut drm_gem_object>,
    gem_prime_import_sg_table: Option<unsafe extern "C" fn(*mut drm_device, *mut dma_buf_attachment, *mut sg_table) -> *mut drm_gem_object>,
    dumb_create: Option<unsafe extern "C" fn(*mut drm_file, *mut drm_device, *mut drm_mode_create_dumb) -> i32>,
    dumb_map_offset: Option<unsafe extern "C" fn(*mut drm_file, *mut drm_device, u32, *mut u64) -> i32>,
    fbdev_probe: Option<unsafe extern "C" fn(*mut drm_fb_helper, *mut drm_fb_helper_surface_size) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut drm_printer, *mut drm_file)>,
    major: i32,
    minor: i32,
    patchlevel: i32,
    name: *mut u8,
    desc: *mut u8,
    driver_features: u32,
    ioctls: *const drm_ioctl_desc,
    num_ioctls: i32,
    fops: *const file_operations,
}


fn drm_core_check_all_features(dev: *const drm_device, features: u32) -> bool {
let supported: u32 = (unsafe { (*(*dev).driver).driver_features }) & unsafe { (*dev).driver_features };
features != 0 && supported & features == features
}

fn drm_core_check_feature(dev: *const drm_device, feature: u32) -> bool {
drm_core_check_all_features(dev, feature as u32)
}

fn drm_drv_uses_atomic_modeset(dev: *mut drm_device) -> bool {
drm_core_check_feature(dev as *const drm_device, 16u32) || !unsafe { (*dev).mode_config.funcs }.is_null() && (unsafe { (*(*dev).mode_config.funcs).atomic_commit }) != None
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_format_info {
    format: u32,
    depth: u8,
    num_planes: u8,
    __slate_anon_3: __SlateRecord_drm_fourcc_h_77,
    block_w: [u8; 4],
    block_h: [u8; 4],
    hsub: u8,
    vsub: u8,
    has_alpha: bool,
    is_yuv: bool,
    is_color_indexed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_framebuffer_funcs {
    destroy: Option<unsafe extern "C" fn(*mut drm_framebuffer)>,
    create_handle: Option<unsafe extern "C" fn(*mut drm_framebuffer, *mut drm_file, *mut u32) -> i32>,
    dirty: Option<unsafe extern "C" fn(*mut drm_framebuffer, *mut drm_file, u32, u32, *mut drm_clip_rect, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_framebuffer {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    comm: [u8; 16],
    format: *const drm_format_info,
    funcs: *const drm_framebuffer_funcs,
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: u64,
    width: u32,
    height: u32,
    flags: i32,
    internal_flags: u32,
    filp_head: list_head,
    obj: [*mut drm_gem_object; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
    secondary: *mut fwnode_handle,
    ops: *const fwnode_operations,
    dev: *mut device,
    suppliers: list_head,
    consumers: list_head,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_reference_args {
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_operations {
    get: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    put: Option<unsafe extern "C" fn(*mut fwnode_handle)>,
    device_is_available: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_match_data: Option<unsafe extern "C" fn(*const fwnode_handle, *const device) -> *const ()>,
    device_dma_supported: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_dma_attr: Option<unsafe extern "C" fn(*const fwnode_handle) -> u32>,
    property_present: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_bool: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_int_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, u32, *mut (), u64) -> i32>,
    property_read_string_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *mut *const u8, u64) -> i32>,
    get_name: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_name_prefix: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_parent: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    get_next_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    get_named_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> *mut fwnode_handle>,
    get_reference_args: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *const u8, u32, u32, *mut fwnode_reference_args) -> i32>,
    graph_get_next_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    graph_get_remote_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    graph_get_port_parent: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    graph_parse_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_endpoint) -> i32>,
    iomap: Option<unsafe extern "C" fn(*mut fwnode_handle, i32) -> *mut ()>,
    irq_get: Option<unsafe extern "C" fn(*const fwnode_handle, u32) -> i32>,
    add_links: Option<unsafe extern "C" fn(*mut fwnode_handle) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iosys_map {
    __slate_anon_0: __SlateRecord_iosys_map_h_110,
    is_iomem: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scanout_buffer {
    format: *const drm_format_info,
    map: [iosys_map; 4],
    pages: *mut *mut page,
    width: u32,
    height: u32,
    pitch: [u32; 4],
    set_pixel: Option<unsafe extern "C" fn(*mut drm_scanout_buffer, u32, u32, u32)>,
    private: *mut (),
}


fn drm_panic_lock(dev: *mut drm_device, flags: u64) {
}

fn drm_panic_unlock(dev: *mut drm_device, flags: u64) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_printer {
    printfn: Option<unsafe extern "C" fn(*mut drm_printer, *mut va_format)>,
    puts: Option<unsafe extern "C" fn(*mut drm_printer, *const u8)>,
    arg: *mut (),
    origin: *const (),
    prefix: *const u8,
    line: __SlateRecord_drm_print_h_181,
    category: u32,
}


fn __drm_to_dev(drm: *const drm_device) -> *mut device {
if !drm.is_null() { unsafe { (*drm).dev } } else { core::ptr::null_mut::<device>() }
}

// Helper for struct drm_device based logging.
// NOTE: this is deprecated in favor of drm_dbg_core(NULL, ...).
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kthread_worker {
    flags: u32,
    lock: raw_spinlock,
    work_list: list_head,
    delayed_work_list: list_head,
    task: *mut task_struct,
    current_work: *mut kthread_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kthread_work {
    node: list_head,
    func: Option<unsafe extern "C" fn(*mut kthread_work)>,
    worker: *mut kthread_worker,
    canceling: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
    sgl: *mut scatterlist,
    nents: u32,
    orig_nents: u32,
}


// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_prime_file_private {
    lock: mutex,
    dmabufs: rb_root,
    handles: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_minor {
    index: i32,
    r#type: i32,
    kdev: *mut device,
    dev: *mut drm_device,
    debugfs_symlink: *mut dentry,
    debugfs_root: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_event {
    completion: *mut completion,
    completion_release: Option<unsafe extern "C" fn(*mut completion)>,
    event: *mut drm_event,
    fence: *mut dma_fence,
    file_priv: *mut drm_file,
    link: list_head,
    pending_link: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_file {
    authenticated: bool,
    stereo_allowed: bool,
    universal_planes: bool,
    atomic: bool,
    aspect_ratio_allowed: bool,
    writeback_connectors: bool,
    plane_color_pipeline: bool,
    was_master: bool,
    is_master: bool,
    supports_virtualized_cursor_plane: bool,
    master: *mut drm_master,
    master_lookup_lock: spinlock,
    pid: *mut pid,
    client_id: u64,
    magic: u32,
    lhead: list_head,
    minor: *mut drm_minor,
    object_idr: idr,
    table_lock: spinlock,
    syncobj_xa: xarray,
    filp: *mut file,
    driver_priv: *mut (),
    fbs: list_head,
    fbs_lock: mutex,
    blobs: list_head,
    event_wait: wait_queue_head,
    pending_event_list: list_head,
    event_list: list_head,
    event_space: i32,
    event_read_lock: mutex,
    prime: drm_prime_file_private,
    client_name: *const u8,
    client_name_lock: mutex,
    debugfs_client: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
    base: drm_pending_event,
    pipe: u32,
    sequence: u64,
    event: __SlateRecord_drm_vblank_h_59,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc_config {
    offdelay_ms: i32,
    disable_immediate: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc_timer {
    timer: hrtimer,
    interval_lock: spinlock,
    interval: i64,
    crtc: *mut drm_crtc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc {
    dev: *mut drm_device,
    queue: wait_queue_head,
    disable_timer: timer_list,
    seqlock: seqlock,
    count: __SlateRecord_types_h_197,
    time: i64,
    refcount: __SlateRecord_types_h_190,
    last: u32,
    max_vblank_count: u32,
    inmodeset: u32,
    pipe: u32,
    framedur_ns: i32,
    linedur_ns: i32,
    hwmode: drm_display_mode,
    config: drm_vblank_crtc_config,
    enabled: bool,
    worker: *mut kthread_worker,
    pending_work: list_head,
    work_wait_queue: wait_queue_head,
    vblank_timer: drm_vblank_crtc_timer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_connector {
    base: drm_connector,
    encoder: drm_encoder,
    pixel_formats_blob_ptr: *mut drm_property_blob,
    job_lock: spinlock,
    job_queue: list_head,
    fence_context: u32,
    fence_lock: spinlock,
    fence_seqno: u64,
    timeline_name: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_job {
    connector: *mut drm_writeback_connector,
    prepared: bool,
    cleanup_work: work_struct,
    list_entry: list_head,
    fb: *mut drm_framebuffer,
    out_fence: *mut dma_fence,
    r#priv: *mut (),
}


/// DOC: overview
///
/// This helper library provides implementations of check and commit functions on
/// top of the CRTC modeset helper callbacks and the plane helper callbacks. It
/// also provides convenience implementations for the atomic state handling
/// callbacks for drivers which don't need to subclass the drm core structures to
/// add their own additional internal state.
///
/// This library also provides default implementations for the check callback in
/// drm_atomic_helper_check() and for the commit callback with
/// drm_atomic_helper_commit(). But the individual stages and callbacks are
/// exposed to allow drivers to mix and match and e.g. use the plane helpers only
/// together with a driver private modeset implementation.
///
/// This library also provides implementations for all the legacy driver
/// interfaces on top of the atomic interface. See drm_atomic_helper_set_config(),
/// drm_atomic_helper_disable_plane(), and the various functions to implement
/// set_property callbacks. New drivers must not implement these functions
/// themselves but must use the provided helpers.
///
/// The atomic helper uses the same function table structures as all other
/// modesetting helpers. See the documentation for &struct drm_crtc_helper_funcs,
/// struct &drm_encoder_helper_funcs and &struct drm_connector_helper_funcs. It
/// also shares the &struct drm_plane_helper_funcs function table with the plane
/// helpers.
fn drm_atomic_helper_plane_changed(state: *mut drm_atomic_commit, old_plane_state: *mut drm_plane_state, plane_state: *mut drm_plane_state, plane: *mut drm_plane) {
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
if !unsafe { (*old_plane_state).crtc }.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*old_plane_state).crtc });
let mut __v0: i64;
let __ret_warn_on: i32 = crtc_state.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("564: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 564b - ., 3; .popsection\n# {0}", const 564 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_plane_changed_1, sym __slate_asm_string_drm_atomic_helper_plane_changed_2, const 92 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("565: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 565b - ., 4; .popsection\n# {0}", const 565 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
unsafe {
let __t0 = true;
(*crtc_state).__slate_bits_0.__set_planes_changed(__t0);
}
}
if !unsafe { (*plane_state).crtc }.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*plane_state).crtc });
let mut __v5: i64;
let __ret_warn_on: i32 = crtc_state.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("566: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 566b - ., 3; .popsection\n# {0}", const 566 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_plane_changed_6, sym __slate_asm_string_drm_atomic_helper_plane_changed_7, const 101 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("567: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 567b - ., 4; .popsection\n# {0}", const 567 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return;
}
unsafe {
let __t1 = true;
(*crtc_state).__slate_bits_0.__set_planes_changed(__t1);
}
}
}

fn handle_conflicting_encoders(state: *mut drm_atomic_commit, disable_conflicting_encoders: bool) -> i32 {
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(crtc_state) as *mut *mut drm_crtc_state;
let mut new_encoder: core::mem::MaybeUninit<*mut drm_encoder> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut drm_encoder = core::ptr::addr_of_mut!(new_encoder) as *mut *mut drm_encoder;
let mut funcs: *const drm_connector_helper_funcs = unsafe { core::mem::zeroed() };
let mut __v11: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut encoder_mask: u32 = unsafe { core::mem::zeroed() };
let mut encoder: core::mem::MaybeUninit<*mut drm_encoder> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut drm_encoder = core::ptr::addr_of_mut!(encoder) as *mut *mut drm_encoder;
let mut conn_iter: core::mem::MaybeUninit<drm_connector_list_iter> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut drm_connector_list_iter = core::ptr::addr_of_mut!(conn_iter) as *mut drm_connector_list_iter;
let mut connector: core::mem::MaybeUninit<*mut drm_connector> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut drm_connector = core::ptr::addr_of_mut!(connector) as *mut *mut drm_connector;
let mut new_conn_state: core::mem::MaybeUninit<*mut drm_connector_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut drm_connector_state = core::ptr::addr_of_mut!(new_conn_state) as *mut *mut drm_connector_state;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_4 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
encoder_mask = 0;
let __slate_auto_bytes_5 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
ret = 0;
// First loop, find all newly assigned encoders from the connectors
// part of the state. If the same encoder is assigned to multiple
// connectors bail out.
*__slate_slot_13 = 0;
while *__slate_slot_13 < (*state).num_connector {
if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(*__slate_slot_13 as isize)).ptr }.is_null() {
*__slate_slot_17 = (*unsafe { (*state).connectors }.wrapping_offset(*__slate_slot_13 as isize)).ptr;
*__slate_slot_17;
*__slate_slot_18 = (*unsafe { (*state).connectors }.wrapping_offset(*__slate_slot_13 as isize)).new_state;
*__slate_slot_18;
__v11 = true;
} else {
__v11 = false;
}
if __v11 {
funcs = (*(*__slate_slot_17)).helper_private;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !unsafe { (*(*__slate_slot_18)).crtc }.is_null() {
if (*funcs).atomic_best_encoder != None {
*__slate_slot_8 = unsafe { (*funcs).atomic_best_encoder }.unwrap()(*__slate_slot_17, state);
} else {
if (*funcs).best_encoder != None {
*__slate_slot_8 = unsafe { (*funcs).best_encoder }.unwrap()(*__slate_slot_17);
} else {
*__slate_slot_8 = drm_connector_get_single_encoder(*__slate_slot_17);
}
}
if !(*__slate_slot_8).is_null() {
if encoder_mask & drm_encoder_mask(*__slate_slot_8 as *const drm_encoder) != 0 {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*__slate_slot_17)).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] on [CONNECTOR:%d:%s] already assigned\n".as_ptr() as *const u8, (*(*__slate_slot_8)).base.id, (*(*__slate_slot_8)).name, (*(*__slate_slot_17)).base.id, (*(*__slate_slot_17)).name);
return 22i32.wrapping_neg();
} else {
encoder_mask |= drm_encoder_mask(*__slate_slot_8 as *const drm_encoder);
}
}
}
}
let __v4: i32 = *__slate_slot_13;
*__slate_slot_13 = __v4.wrapping_add(1);
}
if encoder_mask == 0 {
return 0;
} else {
// Second loop, iterate over all connectors not part of the state.
//
// If a conflicting encoder is found and disable_conflicting_encoders
// is not set, an error is returned. Userspace can provide a solution
// through the atomic ioctl.
//
// If the flag is set conflicting connectors are removed from the CRTC
// and the CRTC is disabled if no encoder is left. This preserves
// compatibility with the legacy set_config behavior.
drm_connector_list_iter_begin((*state).dev, core::ptr::addr_of_mut!(*__slate_slot_16));
'__join_0: {
'__join_8: {
loop {
let __v0: *mut drm_connector = drm_connector_list_iter_next(core::ptr::addr_of_mut!(*__slate_slot_16));
*__slate_slot_17 = __v0;
if !__v0.is_null() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if drm_atomic_get_new_connector_state(state as *const drm_atomic_commit, *__slate_slot_17).is_null() {
*__slate_slot_15 = (*(*(*__slate_slot_17)).state).best_encoder;
if (*__slate_slot_15).is_null() {
__v1 = true;
} else {
__v1 = encoder_mask & drm_encoder_mask(*__slate_slot_15 as *const drm_encoder) == 0;
}
if !__v1 {
if !disable_conflicting_encoders {
break '__join_8;
} else {
*__slate_slot_18 = drm_atomic_get_connector_state(state, *__slate_slot_17);
if IS_ERR(*__slate_slot_18 as *const ()) {
break;
} else {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*__slate_slot_17)).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] in use on [CRTC:%d:%s], disabling [CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*(*__slate_slot_15)).base.id, (*(*__slate_slot_15)).name, (*(*(*__slate_slot_18)).crtc).base.id, (*(*(*__slate_slot_18)).crtc).name, (*(*__slate_slot_17)).base.id, (*(*__slate_slot_17)).name);
*__slate_slot_2 = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, (*(*__slate_slot_18)).crtc);
ret = drm_atomic_set_crtc_for_connector(*__slate_slot_18, core::ptr::null_mut::<drm_crtc>());
if ret != 0 {
break '__join_0;
} else {
if (*(*__slate_slot_2)).connector_mask == 0 {
ret = drm_atomic_set_mode_prop_for_crtc(*__slate_slot_2, core::ptr::null_mut::<drm_property_blob>());
if ret < 0 {
break '__join_0;
} else {
(*(*__slate_slot_2)).active = false;
}
}
}
}
}
}
}
} else {
break '__join_0;
}
}
ret = PTR_ERR(*__slate_slot_18 as *const ()) as i32;
break '__join_0;
}
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*__slate_slot_17)).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] in use on [CRTC:%d:%s] by [CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*(*__slate_slot_15)).base.id, (*(*__slate_slot_15)).name, (*(*(*(*__slate_slot_17)).state).crtc).base.id, (*(*(*(*__slate_slot_17)).state).crtc).name, (*(*__slate_slot_17)).base.id, (*(*__slate_slot_17)).name);
ret = 22i32.wrapping_neg();
}
drm_connector_list_iter_end(core::ptr::addr_of_mut!(*__slate_slot_16));
return ret;
}
}
}

fn set_best_encoder(state: *mut drm_atomic_commit, conn_state: *mut drm_connector_state, encoder: *mut drm_encoder) {
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
if !unsafe { (*conn_state).best_encoder }.is_null() {
// Unset the encoder_mask in the old crtc state.
crtc = unsafe { (*(*(*conn_state).connector).state).crtc };
// A NULL crtc is an error here because we should have
// duplicated a NULL best_encoder when crtc was NULL.
// As an exception restoring duplicated atomic state
// during resume is allowed, so don't warn when
// best_encoder is equal to encoder we intend to set.
let mut __v0: i64;
let __ret_warn_on: i32 = (crtc.is_null() && encoder != unsafe { (*conn_state).best_encoder }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("568: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 568b - ., 3; .popsection\n# {0}", const 568 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_set_best_encoder_1, sym __slate_asm_string_set_best_encoder_2, const 239 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("569: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 569b - ., 4; .popsection\n# {0}", const 569 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if !crtc.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, crtc);
let __v5: u32 = !drm_encoder_mask((unsafe { (*conn_state).best_encoder }) as *const drm_encoder);
unsafe {
(*crtc_state).encoder_mask = (*crtc_state).encoder_mask & __v5;
}
}
}
if !encoder.is_null() {
crtc = unsafe { (*conn_state).crtc };
let mut __v9: i64;
let __ret_warn_on: i32 = crtc.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("570: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 570b - ., 3; .popsection\n# {0}", const 570 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_set_best_encoder_10, sym __slate_asm_string_set_best_encoder_11, const 250 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("571: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 571b - ., 4; .popsection\n# {0}", const 571 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v9 = (__ret_warn_on != 0) as i64;
if !crtc.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, crtc);
let __v14: u32 = drm_encoder_mask(encoder as *const drm_encoder);
unsafe {
(*crtc_state).encoder_mask = (*crtc_state).encoder_mask | __v14;
}
}
}
unsafe {
(*conn_state).best_encoder = encoder;
}
}

fn steal_encoder(state: *mut drm_atomic_commit, encoder: *mut drm_encoder) {
let mut old_connector_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_connector_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if (unsafe { (*new_connector_state).best_encoder }) == encoder {
let encoder_crtc: *mut drm_crtc = unsafe { (*old_connector_state).crtc };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] in use on [CRTC:%d:%s], stealing it\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name, (*encoder_crtc).base.id, (*encoder_crtc).name) };
set_best_encoder(state, new_connector_state, core::ptr::null_mut::<drm_encoder>());
let crtc_state: *mut drm_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, encoder_crtc);
unsafe {
let __t0 = true;
(*crtc_state).__slate_bits_0.__set_connectors_changed(__t0);
}
return;
}
}
i = i.wrapping_add(1);
}
}

fn update_connector_routing(state: *mut drm_atomic_commit, connector: *mut drm_connector, old_connector_state: *mut drm_connector_state, new_connector_state: *mut drm_connector_state, added_by_user: bool) -> i32 {
let mut new_encoder: *mut drm_encoder = unsafe { core::mem::zeroed() };
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"Updating routing for [CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
if (unsafe { (*old_connector_state).crtc }) != unsafe { (*new_connector_state).crtc } {
if !unsafe { (*old_connector_state).crtc }.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*old_connector_state).crtc });
unsafe {
let __t0 = true;
(*crtc_state).__slate_bits_0.__set_connectors_changed(__t0);
}
}
if !unsafe { (*new_connector_state).crtc }.is_null() {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_connector_state).crtc });
unsafe {
let __t1 = true;
(*crtc_state).__slate_bits_0.__set_connectors_changed(__t1);
}
}
}
if unsafe { (*new_connector_state).crtc }.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"Disabling [CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
set_best_encoder(state, new_connector_state, core::ptr::null_mut::<drm_encoder>());
return 0;
}
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_connector_state).crtc });
// For compatibility with legacy users, we want to make sure that
// we allow DPMS On->Off modesets on unregistered connectors. Modesets
// which would result in anything else must be considered invalid, to
// avoid turning on new displays on dead connectors.
//
// Since the connector can be unregistered at any point during an
// atomic check or commit, this is racy. But that's OK: all we care
// about is ensuring that userspace can't do anything but shut off the
// display on a connector that was destroyed after it's been notified,
// not before.
//
// Additionally, we also want to ignore connector registration when
// we're trying to restore an atomic state during system resume since
// there's a chance the connector may have been destroyed during the
// process, but it's better to ignore that then cause
// drm_atomic_helper_resume() to fail.
//
// Last, we want to ignore connector registration when the connector
// was not pulled in the atomic state by user-space (ie, was pulled
// in by the driver, e.g. when updating a DP-MST stream).
let __v0: bool = ((unsafe { (*state).__slate_bits_0.__get_duplicated() }) as i32) == 0 && drm_connector_is_unregistered(connector);
if __v0 && added_by_user && unsafe { (*crtc_state).active } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] is not registered\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
return 22i32.wrapping_neg();
}
let funcs: *const drm_connector_helper_funcs = unsafe { (*connector).helper_private };
if (unsafe { (*funcs).atomic_best_encoder }) != None {
new_encoder = unsafe { unsafe { (*funcs).atomic_best_encoder }.unwrap()(connector, state) };
} else {
if (unsafe { (*funcs).best_encoder }) != None {
new_encoder = unsafe { unsafe { (*funcs).best_encoder }.unwrap()(connector) };
} else {
new_encoder = unsafe { drm_connector_get_single_encoder(connector) };
}
}
if new_encoder.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"No suitable encoder found for [CONNECTOR:%d:%s]\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
return 22i32.wrapping_neg();
}
if !drm_encoder_crtc_ok(new_encoder, unsafe { (*new_connector_state).crtc }) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] incompatible with [CRTC:%d:%s]\n".as_ptr() as *const u8, (*new_encoder).base.id, (*new_encoder).name, (*(*new_connector_state).crtc).base.id, (*(*new_connector_state).crtc).name) };
return 22i32.wrapping_neg();
}
if new_encoder == unsafe { (*new_connector_state).best_encoder } {
set_best_encoder(state, new_connector_state, new_encoder);
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] keeps [ENCODER:%d:%s], now on [CRTC:%d:%s]\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*new_encoder).base.id, (*new_encoder).name, (*(*new_connector_state).crtc).base.id, (*(*new_connector_state).crtc).name) };
return 0;
}
steal_encoder(state, new_encoder);
set_best_encoder(state, new_connector_state, new_encoder);
unsafe {
let __t2 = true;
(*crtc_state).__slate_bits_0.__set_connectors_changed(__t2);
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] using [ENCODER:%d:%s] on [CRTC:%d:%s]\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name, (*new_encoder).base.id, (*new_encoder).name, (*(*new_connector_state).crtc).base.id, (*(*new_connector_state).crtc).name) };
0
}

fn mode_fixup(state: *mut drm_atomic_commit) -> i32 {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if !(((unsafe { (*new_crtc_state).__slate_bits_0.__get_mode_changed() }) as i32) == 0 && ((unsafe { (*new_crtc_state).__slate_bits_0.__get_connectors_changed() }) as i32) == 0) {
unsafe { drm_mode_copy(core::ptr::addr_of_mut!((*new_crtc_state).adjusted_mode), core::ptr::addr_of_mut!((*new_crtc_state).mode) as *const drm_display_mode) };
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v5: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v5 {
let mut __v7: i64;
let __ret_warn_on: i32 = ((!unsafe { (*new_conn_state).best_encoder }.is_null() as i32) != (!unsafe { (*new_conn_state).crtc }.is_null() as i32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("572: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 572b - ., 3; .popsection\n# {0}", const 572 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mode_fixup_8, sym __slate_asm_string_mode_fixup_9, const 442 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("573: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 573b - ., 4; .popsection\n# {0}", const 573 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
if !(unsafe { (*new_conn_state).crtc }.is_null() || unsafe { (*new_conn_state).best_encoder }.is_null()) {
new_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_conn_state).crtc });
// Each encoder has at most one connector (since we always steal
// it away), so we won't call ->mode_fixup twice.
let encoder: *mut drm_encoder = unsafe { (*new_conn_state).best_encoder };
let funcs: *const drm_encoder_helper_funcs = unsafe { (*encoder).helper_private };
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
ret = unsafe { drm_atomic_bridge_chain_check(bridge, new_crtc_state, new_conn_state) };
unsafe { drm_bridge_put(bridge) };
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"Bridge atomic check failed\n".as_ptr() as *const u8) };
return ret;
}
if !funcs.is_null() && (unsafe { (*funcs).atomic_check }) != None {
ret = unsafe { unsafe { (*funcs).atomic_check }.unwrap()(encoder, new_crtc_state, new_conn_state) };
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] check failed\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
return ret;
}
} else {
if !funcs.is_null() && (unsafe { (*funcs).mode_fixup }) != None {
ret = (unsafe { unsafe { (*funcs).mode_fixup }.unwrap()(encoder, core::ptr::addr_of_mut!((*new_crtc_state).mode) as *const drm_display_mode, core::ptr::addr_of_mut!((*new_crtc_state).adjusted_mode)) }) as i32;
if ret == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] fixup failed\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
return 22i32.wrapping_neg();
}
}
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v15: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v15 {
if unsafe { (*new_crtc_state).enable } {
if !(((unsafe { (*new_crtc_state).__slate_bits_0.__get_mode_changed() }) as i32) == 0 && ((unsafe { (*new_crtc_state).__slate_bits_0.__get_connectors_changed() }) as i32) == 0) {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if !(funcs.is_null() || (unsafe { (*funcs).mode_fixup }) == None) {
ret = (unsafe { unsafe { (*funcs).mode_fixup }.unwrap()(crtc, core::ptr::addr_of_mut!((*new_crtc_state).mode) as *const drm_display_mode, core::ptr::addr_of_mut!((*new_crtc_state).adjusted_mode)) }) as i32;
if ret == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*crtc).dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] fixup failed\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
return 22i32.wrapping_neg();
}
}
}
}
}
i = i.wrapping_add(1);
}
0
}

fn mode_valid_path(connector: *mut drm_connector, encoder: *mut drm_encoder, crtc: *mut drm_crtc, mode: *const drm_display_mode) -> i32 {
let mut ret: i32 = unsafe { drm_encoder_mode_valid(encoder, mode) };
if (ret as i32) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"[ENCODER:%d:%s] mode_valid() failed\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
return ret;
}
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
ret = unsafe { drm_bridge_chain_mode_valid(bridge, core::ptr::addr_of_mut!((*connector).display_info) as *const drm_display_info, mode) };
unsafe { drm_bridge_put(bridge) };
if (ret as i32) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"[BRIDGE] mode_valid() failed\n".as_ptr() as *const u8) };
return ret;
}
ret = unsafe { drm_crtc_mode_valid(crtc, mode) };
if (ret as i32) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] mode_valid() failed\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
return ret;
}
ret
}

fn mode_valid(state: *mut drm_atomic_commit) -> i32 {
let mut conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut connector: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let encoder: *mut drm_encoder = unsafe { (*conn_state).best_encoder };
let crtc: *mut drm_crtc = unsafe { (*conn_state).crtc };
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut mode_status: i32 = 0;
if !(crtc.is_null() || encoder.is_null()) {
crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, crtc);
if !crtc_state.is_null() {
if !(((unsafe { (*crtc_state).__slate_bits_0.__get_mode_changed() }) as i32) == 0 && ((unsafe { (*crtc_state).__slate_bits_0.__get_connectors_changed() }) as i32) == 0) {
let mode: *const drm_display_mode = (unsafe { core::ptr::addr_of_mut!((*crtc_state).mode) }) as *const drm_display_mode;
mode_status = mode_valid_path(connector, encoder, crtc, mode);
if (mode_status as i32) != 0 {
return 22i32.wrapping_neg();
}
}
}
}
}
i = i.wrapping_add(1);
}
0
}

fn drm_atomic_check_valid_clones(state: *mut drm_atomic_commit, crtc: *mut drm_crtc) -> i32 {
let mut drm_enc: *mut drm_encoder = unsafe { core::mem::zeroed() };
let crtc_state: *mut drm_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, crtc);
drm_enc = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*(*crtc).dev).mode_config.encoder_list) });
while list_entry_is_head!(drm_enc, unsafe { core::ptr::addr_of_mut!((*(*crtc).dev).mode_config.encoder_list) }, head) == 0 {
if (unsafe { (*crtc_state).encoder_mask }) & drm_encoder_mask(drm_enc as *const drm_encoder) != 0 {
if (unsafe { (*drm_enc).possible_clones }) == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 0u32, c"enc%d possible_clones is 0\n".as_ptr() as *const u8, (*drm_enc).base.id) };
} else {
if (unsafe { (*crtc_state).encoder_mask }) & unsafe { (*drm_enc).possible_clones } != unsafe { (*crtc_state).encoder_mask } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), core::ptr::null::<device>(), 0u32, c"crtc%d failed valid clone check for mask 0x%x\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc_state).encoder_mask) };
return 22i32.wrapping_neg();
}
}
}
drm_enc = list_next_entry__2!(drm_enc, head);
}
0
}

/// drm_atomic_helper_check_modeset - validate state object for modeset changes
/// @dev: DRM device
/// @state: the driver state object
///
/// Check the state object to see if the requested state is physically possible.
/// This does all the CRTC and connector related computations for an atomic
/// update and adds any additional connectors needed for full modesets. It calls
/// the various per-object callbacks in the follow order:
///
/// 1. &drm_connector_helper_funcs.atomic_best_encoder for determining the new encoder.
/// 2. &drm_connector_helper_funcs.atomic_check to validate the connector state.
/// 3. If it's determined a modeset is needed then all connectors on the affected
///    CRTC are added and &drm_connector_helper_funcs.atomic_check is run on them.
/// 4. &drm_encoder_helper_funcs.mode_valid, &drm_bridge_funcs.mode_valid and
///    &drm_crtc_helper_funcs.mode_valid are called on the affected components.
/// 5. &drm_bridge_funcs.mode_fixup is called on all encoder bridges.
/// 6. &drm_encoder_helper_funcs.atomic_check is called to validate any encoder state.
///    This function is only called when the encoder will be part of a configured CRTC,
///    it must not be used for implementing connector property validation.
///    If this function is NULL, &drm_atomic_encoder_helper_funcs.mode_fixup is called
///    instead.
/// 7. &drm_crtc_helper_funcs.mode_fixup is called last, to fix up the mode with CRTC constraints.
///
/// &drm_crtc_state.mode_changed is set when the input mode is changed.
/// &drm_crtc_state.connectors_changed is set when a connector is added or
/// removed from the CRTC.  &drm_crtc_state.active_changed is set when
/// &drm_crtc_state.active changes, which is used for DPMS.
/// &drm_crtc_state.no_vblank is set from the result of drm_dev_has_vblank().
/// See also: drm_atomic_crtc_needs_modeset()
///
/// IMPORTANT:
///
/// Drivers which set &drm_crtc_state.mode_changed (e.g. in their
/// &drm_plane_helper_funcs.atomic_check hooks if a plane update can't be done
/// without a full modeset) _must_ call this function after that change. It is
/// permitted to call this function multiple times for the same update, e.g.
/// when the &drm_crtc_helper_funcs.atomic_check functions depend upon the
/// adjusted dotclock for fifo space allocation and watermark computation.
///
/// RETURNS:
/// Zero for success or -errno
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_check_modeset")]
extern "C" fn drm_atomic_helper_check_modeset(dev: *mut drm_device, state: *mut drm_atomic_commit) -> i32 {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut connector: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut old_connector_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_connector_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i32 = 0;
let mut connectors_mask: u32 = 0;
let mut user_connectors_mask: u32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let __v3: u32 = user_connectors_mask;
user_connectors_mask = ((__v3 as u64) | BIT!(i)) as u32;
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v8: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v8 {
let has_connectors: bool = (unsafe { (*new_crtc_state).connector_mask }) != 0;
let mut __v11: i64;
let __ret_warn_on: i32 = !(!(!drm_modeset_is_locked(unsafe { core::ptr::addr_of_mut!((*crtc).mutex) }))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("574: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 574b - ., 3; .popsection\n# {0}", const 574 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_check_modeset_12, sym __slate_asm_string_drm_atomic_helper_check_modeset_13, const 667 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("575: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 575b - ., 4; .popsection\n# {0}", const 575 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v11 = (__ret_warn_on != 0) as i64;
if !unsafe { drm_mode_equal(core::ptr::addr_of_mut!((*old_crtc_state).mode) as *const drm_display_mode, core::ptr::addr_of_mut!((*new_crtc_state).mode) as *const drm_display_mode) } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] mode changed\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
unsafe {
let __t0 = true;
(*new_crtc_state).__slate_bits_0.__set_mode_changed(__t0);
}
}
if ((unsafe { (*old_crtc_state).enable }) as i32) != ((unsafe { (*new_crtc_state).enable }) as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] enable changed\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
// For clarity this assignment is done here, but
// enable == 0 is only true when there are no
// connectors and a NULL mode.
//
// The other way around is true as well. enable != 0
// implies that connectors are attached and a mode is set.
unsafe {
let __t1 = true;
(*new_crtc_state).__slate_bits_0.__set_mode_changed(__t1);
}
unsafe {
let __t2 = true;
(*new_crtc_state).__slate_bits_0.__set_connectors_changed(__t2);
}
}
if ((unsafe { (*old_crtc_state).active }) as i32) != ((unsafe { (*new_crtc_state).active }) as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] active changed\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
unsafe {
let __t3 = true;
(*new_crtc_state).__slate_bits_0.__set_active_changed(__t3);
}
}
if ((unsafe { (*new_crtc_state).enable }) as i32) != (has_connectors as i32) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] enabled/connectors mismatch (%d/%d)\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name, (*new_crtc_state).enable as i32, has_connectors as i32) };
return 22i32.wrapping_neg();
}
if unsafe { drm_dev_has_vblank(dev as *const drm_device) } {
unsafe {
(*new_crtc_state).no_vblank = false;
}
} else {
unsafe {
(*new_crtc_state).no_vblank = true;
}
}
}
i = i.wrapping_add(1);
}
ret = handle_conflicting_encoders(state, false);
if ret != 0 {
return ret;
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v19: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v19 {
let funcs: *const drm_connector_helper_funcs = unsafe { (*connector).helper_private };
let mut __v22: i64;
let __ret_warn_on: i32 = !(!(!drm_modeset_is_locked(unsafe { core::ptr::addr_of_mut!((*dev).mode_config.connection_mutex) }))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("576: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 576b - ., 3; .popsection\n# {0}", const 576 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_check_modeset_23, sym __slate_asm_string_drm_atomic_helper_check_modeset_24, const 718 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("577: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 577b - ., 4; .popsection\n# {0}", const 577 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v22 = (__ret_warn_on != 0) as i64;
// This only sets crtc->connectors_changed for routing changes,
// drivers must set crtc->connectors_changed themselves when
// connector properties need to be updated.
ret = update_connector_routing(state, connector, old_connector_state, new_connector_state, BIT!(i) & (user_connectors_mask as u64) != 0);
if ret != 0 {
return ret;
}
if !unsafe { (*old_connector_state).crtc }.is_null() {
new_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*old_connector_state).crtc });
if ((unsafe { (*old_connector_state).link_status }) as u32) != ((unsafe { (*new_connector_state).link_status }) as u32) {
unsafe {
let __t4 = true;
(*new_crtc_state).__slate_bits_0.__set_connectors_changed(__t4);
}
}
if ((unsafe { (*old_connector_state).max_requested_bpc }) as i32) != ((unsafe { (*new_connector_state).max_requested_bpc }) as i32) {
unsafe {
let __t5 = true;
(*new_crtc_state).__slate_bits_0.__set_connectors_changed(__t5);
}
}
if ((unsafe { (*old_connector_state).color_format }) as u32) != ((unsafe { (*new_connector_state).color_format }) as u32) {
unsafe {
let __t6 = true;
(*new_crtc_state).__slate_bits_0.__set_connectors_changed(__t6);
}
}
}
if (unsafe { (*funcs).atomic_check }) != None {
ret = unsafe { unsafe { (*funcs).atomic_check }.unwrap()(connector, state) };
}
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] driver check failed\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
return ret;
}
let __v27: u32 = connectors_mask;
connectors_mask = ((__v27 as u64) | BIT!(i)) as u32;
}
i = i.wrapping_add(1);
}
// After all the routing has been prepared we need to add in any
// connector which is itself unchanged, but whose CRTC changes its
// configuration. This must be done before calling mode_fixup in case a
// crtc only changed its mode but has the same set of connectors.
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v32: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v32 {
if drm_atomic_crtc_needs_modeset(new_crtc_state as *const drm_crtc_state) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] needs all connectors, enable: %c, active: %c\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name, if (*new_crtc_state).enable { 121i32 } else { 110i32 }, if (*new_crtc_state).active { 121i32 } else { 110i32 }) };
ret = unsafe { drm_atomic_add_affected_connectors(state, crtc) };
if ret != 0 {
return ret;
}
ret = unsafe { drm_atomic_add_affected_planes(state, crtc) };
if ret != 0 {
return ret;
}
ret = drm_atomic_check_valid_clones(state, crtc);
if ret != 0 {
return ret;
}
}
}
i = i.wrapping_add(1);
}
// Iterate over all connectors again, to make sure atomic_check()
// has been called on them when a modeset is forced.
i = 0;
while i < unsafe { (*state).num_connector } {
let __v38: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v38 {
let funcs: *const drm_connector_helper_funcs = unsafe { (*connector).helper_private };
if (connectors_mask as u64) & BIT!(i) == 0 {
if (unsafe { (*funcs).atomic_check }) != None {
ret = unsafe { unsafe { (*funcs).atomic_check }.unwrap()(connector, state) };
}
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] driver check failed\n".as_ptr() as *const u8, (*connector).base.id, (*connector).name) };
return ret;
}
}
}
i = i.wrapping_add(1);
}
// Iterate over all connectors again, and add all affected bridges to
// the state.
i = 0;
while i < unsafe { (*state).num_connector } {
let __v44: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v44 {
let mut encoder: *mut drm_encoder = unsafe { core::mem::zeroed() };
encoder = unsafe { (*old_connector_state).best_encoder };
ret = unsafe { drm_atomic_add_encoder_bridges(state, encoder) };
if ret != 0 {
return ret;
}
encoder = unsafe { (*new_connector_state).best_encoder };
ret = unsafe { drm_atomic_add_encoder_bridges(state, encoder) };
if ret != 0 {
return ret;
}
}
i = i.wrapping_add(1);
}
ret = mode_valid(state);
if ret != 0 {
return ret;
}
mode_fixup(state)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_check_modeset_578: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_check_modeset)) };

/// drm_atomic_helper_check_wb_connector_state() - Check writeback connector state
/// @connector: corresponding connector
/// @state: the driver state object
///
/// Checks if the writeback connector state is valid, and returns an error if it
/// isn't.
///
/// RETURNS:
/// Zero for success or -errno
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_check_wb_connector_state")]
extern "C" fn drm_atomic_helper_check_wb_connector_state(connector: *mut drm_connector, state: *mut drm_atomic_commit) -> i32 {
let conn_state: *mut drm_connector_state = drm_atomic_get_new_connector_state(state as *const drm_atomic_commit, connector);
let wb_job: *mut drm_writeback_job = unsafe { (*conn_state).writeback_job };
let mut fb: *mut drm_framebuffer = unsafe { core::mem::zeroed() };
let mut nformats: u64 = 0;
if wb_job.is_null() || unsafe { (*wb_job).fb }.is_null() {
return 0;
}
let pixel_format_blob: *mut drm_property_blob = unsafe { (*(*wb_job).connector).pixel_formats_blob_ptr };
nformats = (unsafe { (*pixel_format_blob).length }) / (core::mem::size_of::<u32>() as u64);
let formats: *mut u32 = (unsafe { (*pixel_format_blob).data }) as *mut u32;
fb = unsafe { (*wb_job).fb };
for i in 0..nformats {
if (unsafe { (*(*fb).format).format }) == unsafe { *formats.wrapping_offset(i as isize) } {
return 0;
}
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*connector).dev as *const drm_device) as *const device, 2u32, c"Invalid pixel format %p4cc\n".as_ptr() as *const u8, core::ptr::addr_of!((*(*fb).format).format)) };
22i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_check_wb_connector_state_579: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_check_wb_connector_state)) };

/// drm_atomic_helper_check_plane_state() - Check plane state for validity
/// @plane_state: plane state to check
/// @crtc_state: CRTC state to check
/// @min_scale: minimum @src:@dest scaling factor in 16.16 fixed point
/// @max_scale: maximum @src:@dest scaling factor in 16.16 fixed point
/// @can_position: is it legal to position the plane such that it
///                doesn't cover the entire CRTC?  This will generally
///                only be false for primary planes.
/// @can_update_disabled: can the plane be updated while the CRTC
///                       is disabled?
///
/// Checks that a desired plane update is valid, and updates various
/// bits of derived state (clipped coordinates etc.). Drivers that provide
/// their own plane handling rather than helper-provided implementations may
/// still wish to call this function to avoid duplication of error checking
/// code.
///
/// RETURNS:
/// Zero if update appears valid, error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_check_plane_state")]
extern "C" fn drm_atomic_helper_check_plane_state(plane_state: *mut drm_plane_state, crtc_state: *const drm_crtc_state, min_scale: i32, max_scale: i32, can_position: bool, can_update_disabled: bool) -> i32 {
let fb: *mut drm_framebuffer = unsafe { (*plane_state).fb };
let src: *mut drm_rect = unsafe { core::ptr::addr_of_mut!((*plane_state).src) };
let dst: *mut drm_rect = unsafe { core::ptr::addr_of_mut!((*plane_state).dst) };
let rotation: u32 = unsafe { (*plane_state).rotation };
let mut clip: drm_rect = {
let mut __t0: drm_rect = unsafe { core::mem::zeroed() };
__t0
};
let mut hscale: i32 = 0;
let mut vscale: i32 = 0;
let mut __v0: i64;
{
let __ret_warn_on: i32 = (!unsafe { (*plane_state).crtc }.is_null() && (unsafe { (*plane_state).crtc }) != unsafe { (*crtc_state).crtc }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("580: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 580b - ., 3; .popsection\n# {0}", const 580 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_check_plane_state_1, sym __slate_asm_string_drm_atomic_helper_check_plane_state_2, const 913 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("581: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 581b - ., 4; .popsection\n# {0}", const 581 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
unsafe {
*src = drm_plane_state_src(plane_state as *const drm_plane_state);
}
unsafe {
*dst = drm_plane_state_dest(plane_state as *const drm_plane_state);
}
if fb.is_null() {
unsafe {
(*plane_state).visible = false;
}
return 0;
}
// crtc should only be NULL when disabling (i.e., !fb)
let mut __v5: i64;
let __ret_warn_on: i32 = unsafe { (*plane_state).crtc }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("582: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 582b - ., 3; .popsection\n# {0}", const 582 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_check_plane_state_6, sym __slate_asm_string_drm_atomic_helper_check_plane_state_7, const 924 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("583: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 583b - ., 4; .popsection\n# {0}", const 583 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
unsafe {
(*plane_state).visible = false;
}
return 0;
}
if !unsafe { (*crtc_state).enable } && !can_update_disabled {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*plane_state).plane).dev as *const drm_device) as *const device, 2u32, c"Cannot update plane of a disabled CRTC.\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
unsafe { drm_rect_rotate(src, ((*fb).width << 16i32) as i32, ((*fb).height << 16i32) as i32, rotation) };
// Check scaling
hscale = unsafe { drm_rect_calc_hscale(src as *const drm_rect, dst as *const drm_rect, min_scale, max_scale) };
vscale = unsafe { drm_rect_calc_vscale(src as *const drm_rect, dst as *const drm_rect, min_scale, max_scale) };
if hscale < 0 || vscale < 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*plane_state).plane).dev as *const drm_device) as *const device, 2u32, c"Invalid scaling of plane\n".as_ptr() as *const u8) };
unsafe { drm_rect_debug_print(c"src: ".as_ptr() as *const u8, core::ptr::addr_of_mut!((*plane_state).src) as *const drm_rect, true) };
unsafe { drm_rect_debug_print(c"dst: ".as_ptr() as *const u8, core::ptr::addr_of_mut!((*plane_state).dst) as *const drm_rect, false) };
return 34i32.wrapping_neg();
}
if unsafe { (*crtc_state).enable } {
unsafe { drm_mode_get_hv_timing(core::ptr::addr_of!((*crtc_state).mode), core::ptr::addr_of_mut!(clip.x2), core::ptr::addr_of_mut!(clip.y2)) };
}
unsafe {
(*plane_state).visible = drm_rect_clip_scaled(src, dst, core::ptr::addr_of_mut!(clip) as *const drm_rect);
}
unsafe { drm_rect_rotate_inv(src, ((*fb).width << 16i32) as i32, ((*fb).height << 16i32) as i32, rotation) };
if !unsafe { (*plane_state).visible } {
return 0;
}
// Plane isn't visible; some drivers can handle this
// so we just return success here.  Drivers that can't
// (including those that use the primary plane helper's
// update function) will return an error from their
// update_plane handler.
let __v10: bool = !can_position && !drm_rect_equals(dst as *const drm_rect, unsafe { &(*(core::ptr::addr_of_mut!(clip) as *const drm_rect)) });
if __v10 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*(*plane_state).plane).dev as *const drm_device) as *const device, 2u32, c"Plane must cover entire CRTC\n".as_ptr() as *const u8) };
unsafe { drm_rect_debug_print(c"dst: ".as_ptr() as *const u8, dst as *const drm_rect, false) };
unsafe { drm_rect_debug_print(c"clip: ".as_ptr() as *const u8, core::ptr::addr_of_mut!(clip) as *const drm_rect, false) };
return 22i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_check_plane_state_584: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_plane_state, *const drm_crtc_state, i32, i32, bool, bool) -> i32>, *mut ()>(Some(drm_atomic_helper_check_plane_state)) };

/// drm_atomic_helper_check_crtc_primary_plane() - Check CRTC state for primary plane
/// @crtc_state: CRTC state to check
///
/// Checks that a CRTC has at least one primary plane attached to it, which is
/// a requirement on some hardware. Note that this only involves the CRTC side
/// of the test. To test if the primary plane is visible or if it can be updated
/// without the CRTC being enabled, use drm_atomic_helper_check_plane_state() in
/// the plane's atomic check.
///
/// RETURNS:
/// 0 if a primary plane is attached to the CRTC, or an error code otherwise
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_check_crtc_primary_plane")]
extern "C" fn drm_atomic_helper_check_crtc_primary_plane(crtc_state: *mut drm_crtc_state) -> i32 {
let crtc: *mut drm_crtc = unsafe { (*crtc_state).crtc };
let dev: *mut drm_device = unsafe { (*crtc).dev };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
// needs at least one primary plane to be enabled
plane = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*dev).mode_config.plane_list) });
while list_entry_is_head!(plane, unsafe { core::ptr::addr_of_mut!((*dev).mode_config.plane_list) }, head) == 0 {
if (unsafe { (*crtc_state).plane_mask }) & drm_plane_mask(plane as *const drm_plane) != 0 {
if ((unsafe { (*plane).r#type }) as u32) == 1 {
return 0;
}
}
plane = list_next_entry__4!(plane, head);
}
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] primary plane missing\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
22i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_check_crtc_primary_plane_585: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_crtc_state) -> i32>, *mut ()>(Some(drm_atomic_helper_check_crtc_primary_plane)) };

/// drm_atomic_helper_check_planes - validate state object for planes changes
/// @dev: DRM device
/// @state: the driver state object
///
/// Check the state object to see if the requested state is physically possible.
/// This does all the plane update related checks using by calling into the
/// &drm_crtc_helper_funcs.atomic_check and &drm_plane_helper_funcs.atomic_check
/// hooks provided by the driver.
///
/// It also sets &drm_crtc_state.planes_changed to indicate that a CRTC has
/// updated planes.
///
/// RETURNS:
/// Zero for success or -errno
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_check_planes")]
extern "C" fn drm_atomic_helper_check_planes(dev: *mut drm_device, state: *mut drm_atomic_commit) -> i32 {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut old_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v0: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let mut __v3: i64;
let __ret_warn_on: i32 = !(!(!drm_modeset_is_locked(unsafe { core::ptr::addr_of_mut!((*plane).mutex) }))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("586: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 586b - ., 3; .popsection\n# {0}", const 586 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_check_planes_4, sym __slate_asm_string_drm_atomic_helper_check_planes_5, const 1037 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("587: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 587b - ., 4; .popsection\n# {0}", const 587 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
drm_atomic_helper_plane_changed(state, old_plane_state, new_plane_state, plane);
unsafe { drm_atomic_helper_check_plane_damage(state, new_plane_state) };
if !(funcs.is_null() || (unsafe { (*funcs).atomic_check }) == None) {
ret = unsafe { unsafe { (*funcs).atomic_check }.unwrap()(plane, state) };
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*plane).dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] atomic driver check failed\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
return ret;
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v11: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
let new_crtc_state: *mut drm_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v11 {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if !(funcs.is_null() || (unsafe { (*funcs).atomic_check }) == None) {
ret = unsafe { unsafe { (*funcs).atomic_check }.unwrap()(crtc, state) };
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*crtc).dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] atomic driver check failed\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
return ret;
}
}
}
i = i.wrapping_add(1);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_check_planes_588: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_check_planes)) };

/// drm_atomic_helper_check - validate state object
/// @dev: DRM device
/// @state: the driver state object
///
/// Check the state object to see if the requested state is physically possible.
/// Only CRTCs and planes have check callbacks, so for any additional (global)
/// checking that a driver needs it can simply wrap that around this function.
/// Drivers without such needs can directly use this as their
/// &drm_mode_config_funcs.atomic_check callback.
///
/// This just wraps the two parts of the state checking for planes and modeset
/// state in the default order: First it calls drm_atomic_helper_check_modeset()
/// and then drm_atomic_helper_check_planes(). The assumption is that the
/// @drm_plane_helper_funcs.atomic_check and @drm_crtc_helper_funcs.atomic_check
/// functions depend upon an updated adjusted_mode.clock to e.g. properly compute
/// watermarks.
///
/// Note that zpos normalization will add all enable planes to the state which
/// might not desired for some drivers.
/// For example enable/disable of a cursor plane which have fixed zpos value
/// would trigger all other enabled planes to be forced to the state change.
///
/// IMPORTANT:
///
/// As this function calls drm_atomic_helper_check_modeset() internally, its
/// restrictions also apply:
/// Drivers which set &drm_crtc_state.mode_changed (e.g. in their
/// &drm_plane_helper_funcs.atomic_check hooks if a plane update can't be done
/// without a full modeset) _must_ call drm_atomic_helper_check_modeset()
/// function again after that change.
///
/// RETURNS:
/// Zero for success or -errno
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_check")]
extern "C" fn drm_atomic_helper_check(dev: *mut drm_device, state: *mut drm_atomic_commit) -> i32 {
let mut ret: i32 = drm_atomic_helper_check_modeset(dev, state);
if ret != 0 {
return ret;
}
if unsafe { (*dev).mode_config.normalize_zpos } {
ret = unsafe { drm_atomic_normalize_zpos(dev, state) };
if ret != 0 {
return ret;
}
}
ret = drm_atomic_helper_check_planes(dev, state);
if ret != 0 {
return ret;
}
if ((unsafe { (*state).__slate_bits_0.__get_legacy_cursor_update() }) as i32) != 0 {
unsafe {
let __t0 = drm_atomic_helper_async_check(dev, state) == 0;
(*state).__slate_bits_0.__set_async_update(__t0);
}
}
unsafe { drm_self_refresh_helper_alter_state(state) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_check_589: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_check)) };

fn crtc_needs_disable(old_state: *mut drm_crtc_state, new_state: *mut drm_crtc_state) -> bool {
// No new_state means the CRTC is off, so the only criteria is whether
// it's currently active or in self refresh mode.
if new_state.is_null() {
return drm_atomic_crtc_effectively_active(old_state as *const drm_crtc_state);
}
// We need to disable bridge(s) and CRTC if we're transitioning out of
// self-refresh and changing CRTCs at the same time, because the
// bridge tracks self-refresh status via CRTC state.
if (unsafe { (*old_state).self_refresh_active }) && (unsafe { (*old_state).crtc }) != unsafe { (*new_state).crtc } {
return true;
}
// We also need to run through the crtc_funcs->disable() function if
// the CRTC is currently on, if it's transitioning to self refresh
// mode, or if it's in self refresh mode and needs to be fully
// disabled.
(unsafe { (*old_state).active }) || (unsafe { (*old_state).self_refresh_active }) && !unsafe { (*new_state).active } || unsafe { (*new_state).self_refresh_active }
}

/// drm_atomic_helper_commit_encoder_bridge_disable - disable bridges and encoder
/// @dev: DRM device
/// @state: the driver state object
///
/// Loops over all connectors in the current state and if the CRTC needs
/// it, disables the bridge chain all the way, then disables the encoder
/// afterwards.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_encoder_bridge_disable")]
extern "C" fn drm_atomic_helper_commit_encoder_bridge_disable(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
// Shut down everything that's in the changeset and currently
// still on. So need to check the old, saved state.
if !unsafe { (*old_conn_state).crtc }.is_null() {
let old_crtc_state: *mut drm_crtc_state = drm_atomic_get_old_crtc_state(state as *const drm_atomic_commit, unsafe { (*old_conn_state).crtc });
if !unsafe { (*new_conn_state).crtc }.is_null() {
new_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_conn_state).crtc });
} else {
new_crtc_state = core::ptr::null_mut::<drm_crtc_state>();
}
let __v4: bool = !crtc_needs_disable(old_crtc_state, new_crtc_state) || !drm_atomic_crtc_needs_modeset((unsafe { (*(*old_conn_state).crtc).state }) as *const drm_crtc_state);
if !__v4 {
let encoder: *mut drm_encoder = unsafe { (*old_conn_state).best_encoder };
// We shouldn't get this far if we didn't previously have
// an encoder.. but WARN_ON() rather than explode.
let mut __v5: i64;
let __ret_warn_on: i32 = encoder.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("590: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 590b - ., 3; .popsection\n# {0}", const 590 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_disable_6, sym __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_disable_7, const 1220 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v5 = (__ret_warn_on != 0) as i64;
if __v5 == 0 {
let funcs: *const drm_encoder_helper_funcs = unsafe { (*encoder).helper_private };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"disabling [ENCODER:%d:%s]\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
// Each encoder has at most one connector (since we always steal
// it away), so we won't call disable hooks twice.
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
unsafe { drm_atomic_bridge_chain_disable(bridge, state) };
unsafe { drm_bridge_put(bridge) };
// Right function depends upon target state.
if !funcs.is_null() {
if (unsafe { (*funcs).atomic_disable }) != None {
unsafe { unsafe { (*funcs).atomic_disable }.unwrap()(encoder, state) };
} else {
if !unsafe { (*new_conn_state).crtc }.is_null() && (unsafe { (*funcs).prepare }) != None {
unsafe { unsafe { (*funcs).prepare }.unwrap()(encoder) };
} else {
if (unsafe { (*funcs).disable }) != None {
unsafe { unsafe { (*funcs).disable }.unwrap()(encoder) };
} else {
if (unsafe { (*funcs).dpms }) != None {
unsafe { unsafe { (*funcs).dpms }.unwrap()(encoder, 3i32) };
}
}
}
}
}
}
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_encoder_bridge_disable_592: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_encoder_bridge_disable)) };

/// drm_atomic_helper_commit_crtc_disable - disable CRTSs
/// @dev: DRM device
/// @state: the driver state object
///
/// Loops over all CRTCs in the current state and if the CRTC needs
/// it, disables it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_crtc_disable")]
extern "C" fn drm_atomic_helper_commit_crtc_disable(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let mut ret: i32 = 0;
// Shut down everything that needs a full modeset.
if drm_atomic_crtc_needs_modeset(new_crtc_state as *const drm_crtc_state) {
if crtc_needs_disable(old_crtc_state, new_crtc_state) {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"disabling [CRTC:%d:%s]\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
// Right function depends upon target state.
if (unsafe { (*new_crtc_state).enable }) && (unsafe { (*funcs).prepare }) != None {
unsafe { unsafe { (*funcs).prepare }.unwrap()(crtc) };
} else {
if (unsafe { (*funcs).atomic_disable }) != None {
unsafe { unsafe { (*funcs).atomic_disable }.unwrap()(crtc, state) };
} else {
if (unsafe { (*funcs).disable }) != None {
unsafe { unsafe { (*funcs).disable }.unwrap()(crtc) };
} else {
if (unsafe { (*funcs).dpms }) != None {
unsafe { unsafe { (*funcs).dpms }.unwrap()(crtc, 3i32) };
}
}
}
}
if unsafe { drm_dev_has_vblank(dev as *const drm_device) } {
ret = unsafe { drm_crtc_vblank_get(crtc) };
// Self-refresh is not a true "disable"; ensure vblank remains
// enabled.
if unsafe { (*new_crtc_state).self_refresh_active } {
let mut __v4: i32;
let __ret_warn_on: i32 = (ret != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"driver disabled vblank in self-refresh\n".as_ptr() as *const u8) };
let mut __v5: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v5 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v6: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_atomic_helper_commit_crtc_disable_7, sym __slate_asm_string_drm_atomic_helper_commit_crtc_disable_8, const 1303 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v6 = bug;
unsafe { __v5.unwrap()(__v6) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v4 = __ret_warn_on;
} else {
let mut __v10: i32;
let __ret_warn_on: i32 = (ret != 22i32.wrapping_neg()) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"driver forgot to call drm_crtc_vblank_off()\n".as_ptr() as *const u8) };
let mut __v11: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v11 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v12: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_drm_atomic_helper_commit_crtc_disable_13, sym __slate_asm_string_drm_atomic_helper_commit_crtc_disable_14, const 1306 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v12 = bug;
unsafe { __v11.unwrap()(__v12) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v10 = __ret_warn_on;
}
if ret == 0 {
unsafe { drm_crtc_vblank_put(crtc) };
}
}
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_593: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_594: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_crtc_disable_595: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_crtc_disable)) };

/// drm_atomic_helper_commit_encoder_bridge_post_disable - post-disable encoder bridges
/// @dev: DRM device
/// @state: the driver state object
///
/// Loops over all connectors in the current state and if the CRTC needs
/// it, post-disables all encoder bridges.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_encoder_bridge_post_disable")]
extern "C" fn drm_atomic_helper_commit_encoder_bridge_post_disable(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
// Shut down everything that's in the changeset and currently
// still on. So need to check the old, saved state.
if !unsafe { (*old_conn_state).crtc }.is_null() {
let old_crtc_state: *mut drm_crtc_state = drm_atomic_get_old_crtc_state(state as *const drm_atomic_commit, unsafe { (*old_conn_state).crtc });
if !unsafe { (*new_conn_state).crtc }.is_null() {
new_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, unsafe { (*new_conn_state).crtc });
} else {
new_crtc_state = core::ptr::null_mut::<drm_crtc_state>();
}
let __v4: bool = !crtc_needs_disable(old_crtc_state, new_crtc_state) || !drm_atomic_crtc_needs_modeset((unsafe { (*(*old_conn_state).crtc).state }) as *const drm_crtc_state);
if !__v4 {
let encoder: *mut drm_encoder = unsafe { (*old_conn_state).best_encoder };
// We shouldn't get this far if we didn't previously have
// an encoder.. but WARN_ON() rather than explode.
let mut __v5: i64;
let __ret_warn_on: i32 = encoder.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("596: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 596b - ., 3; .popsection\n# {0}", const 596 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_post_disable_6, sym __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_post_disable_7, const 1358 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("597: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 597b - ., 4; .popsection\n# {0}", const 597 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 == 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"post-disabling bridges [ENCODER:%d:%s]\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
// Each encoder has at most one connector (since we always steal
// it away), so we won't call disable hooks twice.
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
unsafe { drm_atomic_bridge_chain_post_disable(bridge, state) };
unsafe { drm_bridge_put(bridge) };
}
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_encoder_bridge_post_disable_598: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_encoder_bridge_post_disable)) };

fn disable_outputs(dev: *mut drm_device, state: *mut drm_atomic_commit) {
drm_atomic_helper_commit_encoder_bridge_disable(dev, state);
drm_atomic_helper_commit_encoder_bridge_post_disable(dev, state);
drm_atomic_helper_commit_crtc_disable(dev, state);
}

/// drm_atomic_helper_update_legacy_modeset_state - update legacy modeset state
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This function updates all the various legacy modeset state pointers in
/// connectors, encoders and CRTCs.
///
/// Drivers can use this for building their own atomic commit if they don't have
/// a pure helper-based modeset implementation.
///
/// Since these updates are not synchronized with lockings, only code paths
/// called from &drm_mode_config_helper_funcs.atomic_commit_tail can look at the
/// legacy state filled out by this helper. Defacto this means this helper and
/// the legacy state pointers are only really useful for transitioning an
/// existing driver to the atomic world.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_update_legacy_modeset_state")]
extern "C" fn drm_atomic_helper_update_legacy_modeset_state(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut connector: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
// clear out existing links and update dpms
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if !unsafe { (*connector).encoder }.is_null() {
let mut __v3: i64;
let __ret_warn_on: i32 = unsafe { (*(*connector).encoder).crtc }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("599: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 599b - ., 3; .popsection\n# {0}", const 599 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_4, sym __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_5, const 1415 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("600: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 600b - ., 4; .popsection\n# {0}", const 600 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
unsafe {
(*(*connector).encoder).crtc = core::ptr::null_mut::<drm_crtc>();
}
unsafe {
(*connector).encoder = core::ptr::null_mut::<drm_encoder>();
}
}
crtc = unsafe { (*new_conn_state).crtc };
let __v8: bool = crtc.is_null() && !unsafe { (*old_conn_state).crtc }.is_null() || !crtc.is_null() && drm_atomic_crtc_needs_modeset((unsafe { (*crtc).state }) as *const drm_crtc_state);
if __v8 {
let mut mode: i32 = 3;
if !crtc.is_null() && unsafe { (*(*crtc).state).active } {
mode = 0;
}
unsafe {
(*connector).dpms = mode;
}
}
}
i = i.wrapping_add(1);
}
// set new links
i = 0;
while i < unsafe { (*state).num_connector } {
let __v13: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v13 {
if !unsafe { (*new_conn_state).crtc }.is_null() {
let mut __v16: i64;
let __ret_warn_on: i32 = unsafe { (*new_conn_state).best_encoder }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("601: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 601b - ., 3; .popsection\n# {0}", const 601 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_17, sym __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_18, const 1438 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("602: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 602b - ., 4; .popsection\n# {0}", const 602 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v16 = (__ret_warn_on != 0) as i64;
if __v16 == 0 {
unsafe {
(*connector).encoder = (*new_conn_state).best_encoder;
}
unsafe {
(*(*connector).encoder).crtc = (*new_conn_state).crtc;
}
}
}
}
i = i.wrapping_add(1);
}
// set legacy state in the crtc structure
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v23: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v23 {
let primary: *mut drm_plane = unsafe { (*crtc).primary };
unsafe {
(*crtc).mode = (*new_crtc_state).mode;
}
unsafe {
(*crtc).enabled = (*new_crtc_state).enable;
}
let new_plane_state: *mut drm_plane_state = drm_atomic_get_new_plane_state(state as *const drm_atomic_commit, primary);
if !new_plane_state.is_null() && (unsafe { (*new_plane_state).crtc }) == crtc {
unsafe {
(*crtc).x = ((*new_plane_state).src_x >> 16i32) as i32;
}
unsafe {
(*crtc).y = ((*new_plane_state).src_y >> 16i32) as i32;
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_update_legacy_modeset_state_603: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_update_legacy_modeset_state)) };

/// drm_atomic_helper_calc_timestamping_constants - update vblank timestamping constants
/// @state: atomic state object
///
/// Updates the timestamping constants used for precise vblank timestamps
/// by calling drm_calc_timestamping_constants() for all enabled crtcs in @state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_calc_timestamping_constants")]
extern "C" fn drm_atomic_helper_calc_timestamping_constants(state: *mut drm_atomic_commit) {
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if unsafe { (*new_crtc_state).enable } {
unsafe { drm_calc_timestamping_constants(crtc, core::ptr::addr_of_mut!((*new_crtc_state).adjusted_mode) as *const drm_display_mode) };
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_calc_timestamping_constants_604: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_calc_timestamping_constants)) };

/// drm_atomic_helper_commit_crtc_set_mode - set the new mode
/// @dev: DRM device
/// @state: the driver state object
///
/// Loops over all connectors in the current state and if the mode has
/// changed, change the mode of the CRTC, then call down the bridge
/// chain and change the mode in all bridges as well.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_crtc_set_mode")]
extern "C" fn drm_atomic_helper_commit_crtc_set_mode(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if ((unsafe { (*new_crtc_state).__slate_bits_0.__get_mode_changed() }) as i32) != 0 {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if (unsafe { (*new_crtc_state).enable }) && (unsafe { (*funcs).mode_set_nofb }) != None {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"modeset on [CRTC:%d:%s]\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
unsafe { unsafe { (*funcs).mode_set_nofb }.unwrap()(crtc) };
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v5: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v5 {
if !unsafe { (*new_conn_state).best_encoder }.is_null() {
let encoder: *mut drm_encoder = unsafe { (*new_conn_state).best_encoder };
let funcs: *const drm_encoder_helper_funcs = unsafe { (*encoder).helper_private };
new_crtc_state = unsafe { (*(*new_conn_state).crtc).state };
let mode: *mut drm_display_mode = unsafe { core::ptr::addr_of_mut!((*new_crtc_state).mode) };
let adjusted_mode: *mut drm_display_mode = unsafe { core::ptr::addr_of_mut!((*new_crtc_state).adjusted_mode) };
if !(((unsafe { (*new_crtc_state).__slate_bits_0.__get_mode_changed() }) as i32) == 0 && ((unsafe { (*new_crtc_state).__slate_bits_0.__get_connectors_changed() }) as i32) == 0) {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"modeset on [ENCODER:%d:%s]\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
// Each encoder has at most one connector (since we always steal
// it away), so we won't call mode_set hooks twice.
if !funcs.is_null() && (unsafe { (*funcs).atomic_mode_set }) != None {
unsafe { unsafe { (*funcs).atomic_mode_set }.unwrap()(encoder, new_crtc_state, new_conn_state) };
} else {
if !funcs.is_null() && (unsafe { (*funcs).mode_set }) != None {
unsafe { unsafe { (*funcs).mode_set }.unwrap()(encoder, mode, adjusted_mode) };
}
}
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
unsafe { drm_bridge_chain_mode_set(bridge, mode as *const drm_display_mode, adjusted_mode as *const drm_display_mode) };
unsafe { drm_bridge_put(bridge) };
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_crtc_set_mode_605: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_crtc_set_mode)) };

/// drm_atomic_helper_commit_modeset_disables - modeset commit to disable outputs
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This function shuts down all the outputs that need to be shut down and
/// prepares them (if required) with the new mode.
///
/// For compatibility with legacy CRTC helpers this should be called before
/// drm_atomic_helper_commit_planes(), which is what the default commit function
/// does. But drivers with different needs can group the modeset commits together
/// and do the plane commits at the end. This is useful for drivers doing runtime
/// PM since planes updates then only happen when the CRTC is actually enabled.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_modeset_disables")]
extern "C" fn drm_atomic_helper_commit_modeset_disables(dev: *mut drm_device, state: *mut drm_atomic_commit) {
disable_outputs(dev, state);
drm_atomic_helper_update_legacy_modeset_state(dev, state);
drm_atomic_helper_calc_timestamping_constants(state);
drm_atomic_helper_commit_crtc_set_mode(dev, state);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_modeset_disables_606: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_modeset_disables)) };

/// drm_atomic_helper_commit_writebacks - issue writebacks
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This loops over the connectors, checks if the new state requires
/// a writeback job to be issued and in that case issues an atomic
/// commit on each connector.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_writebacks")]
extern "C" fn drm_atomic_helper_commit_writebacks(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut connector: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let funcs: *const drm_connector_helper_funcs = unsafe { (*connector).helper_private };
if (unsafe { (*funcs).atomic_commit }) != None {
if !unsafe { (*new_conn_state).writeback_job }.is_null() && !unsafe { (*(*new_conn_state).writeback_job).fb }.is_null() {
let mut __v3: i64;
let __ret_warn_on: i32 = ((unsafe { (*connector).connector_type }) != 18) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("607: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 607b - ., 3; .popsection\n# {0}", const 607 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_writebacks_4, sym __slate_asm_string_drm_atomic_helper_commit_writebacks_5, const 1608 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("608: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 608b - ., 4; .popsection\n# {0}", const 608 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
unsafe { unsafe { (*funcs).atomic_commit }.unwrap()(connector, state) };
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_writebacks_609: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_writebacks)) };

/// drm_atomic_helper_commit_encoder_bridge_pre_enable - pre-enable bridges
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This loops over the connectors and if the CRTC needs it, pre-enables
/// the entire bridge chain.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_encoder_bridge_pre_enable")]
extern "C" fn drm_atomic_helper_commit_encoder_bridge_pre_enable(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if !unsafe { (*new_conn_state).best_encoder }.is_null() {
let __v3: bool = !unsafe { (*(*(*new_conn_state).crtc).state).active } || !drm_atomic_crtc_needs_modeset((unsafe { (*(*new_conn_state).crtc).state }) as *const drm_crtc_state);
if !__v3 {
let encoder: *mut drm_encoder = unsafe { (*new_conn_state).best_encoder };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"pre-enabling bridges [ENCODER:%d:%s]\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
// Each encoder has at most one connector (since we always steal
// it away), so we won't call enable hooks twice.
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
unsafe { drm_atomic_bridge_chain_pre_enable(bridge, state) };
unsafe { drm_bridge_put(bridge) };
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_encoder_bridge_pre_enable_610: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_encoder_bridge_pre_enable)) };

/// drm_atomic_helper_commit_crtc_enable - enables the CRTCs
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This loops over CRTCs in the new state, and of the CRTC needs
/// it, enables it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_crtc_enable")]
extern "C" fn drm_atomic_helper_commit_crtc_enable(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
let old_crtc_state: *mut drm_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
// Need to filter out CRTCs where only planes change.
if drm_atomic_crtc_needs_modeset(new_crtc_state as *const drm_crtc_state) {
if unsafe { (*new_crtc_state).active } {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if unsafe { (*new_crtc_state).enable } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"enabling [CRTC:%d:%s]\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
if (unsafe { (*funcs).atomic_enable }) != None {
unsafe { unsafe { (*funcs).atomic_enable }.unwrap()(crtc, state) };
} else {
if (unsafe { (*funcs).commit }) != None {
unsafe { unsafe { (*funcs).commit }.unwrap()(crtc) };
}
}
}
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_crtc_enable_611: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_crtc_enable)) };

/// drm_atomic_helper_commit_encoder_bridge_enable - enables the bridges
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This loops over all connectors in the new state, and of the CRTC needs
/// it, enables the entire bridge chain.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_encoder_bridge_enable")]
extern "C" fn drm_atomic_helper_commit_encoder_bridge_enable(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*state).num_connector } {
let __v0: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
let connector: *mut drm_connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if !unsafe { (*new_conn_state).best_encoder }.is_null() {
let __v3: bool = !unsafe { (*(*(*new_conn_state).crtc).state).active } || !drm_atomic_crtc_needs_modeset((unsafe { (*(*new_conn_state).crtc).state }) as *const drm_crtc_state);
if !__v3 {
let encoder: *mut drm_encoder = unsafe { (*new_conn_state).best_encoder };
let funcs: *const drm_encoder_helper_funcs = unsafe { (*encoder).helper_private };
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"enabling [ENCODER:%d:%s]\n".as_ptr() as *const u8, (*encoder).base.id, (*encoder).name) };
// Each encoder has at most one connector (since we always steal
// it away), so we won't call enable hooks twice.
let bridge: *mut drm_bridge = drm_bridge_chain_get_first_bridge(encoder);
if !funcs.is_null() {
if (unsafe { (*funcs).atomic_enable }) != None {
unsafe { unsafe { (*funcs).atomic_enable }.unwrap()(encoder, state) };
} else {
if (unsafe { (*funcs).enable }) != None {
unsafe { unsafe { (*funcs).enable }.unwrap()(encoder) };
} else {
if (unsafe { (*funcs).commit }) != None {
unsafe { unsafe { (*funcs).commit }.unwrap()(encoder) };
}
}
}
}
unsafe { drm_atomic_bridge_chain_enable(bridge, state) };
unsafe { drm_bridge_put(bridge) };
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_encoder_bridge_enable_612: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_encoder_bridge_enable)) };

/// drm_atomic_helper_commit_modeset_enables - modeset commit to enable outputs
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This function enables all the outputs with the new configuration which had to
/// be turned off for the update.
///
/// For compatibility with legacy CRTC helpers this should be called after
/// drm_atomic_helper_commit_planes(), which is what the default commit function
/// does. But drivers with different needs can group the modeset commits together
/// and do the plane commits at the end. This is useful for drivers doing runtime
/// PM since planes updates then only happen when the CRTC is actually enabled.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_modeset_enables")]
extern "C" fn drm_atomic_helper_commit_modeset_enables(dev: *mut drm_device, state: *mut drm_atomic_commit) {
drm_atomic_helper_commit_crtc_enable(dev, state);
drm_atomic_helper_commit_encoder_bridge_pre_enable(dev, state);
drm_atomic_helper_commit_encoder_bridge_enable(dev, state);
drm_atomic_helper_commit_writebacks(dev, state);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_modeset_enables_613: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_modeset_enables)) };

/// For atomic updates which touch just a single CRTC, calculate the time of the
/// next vblank, and inform all the fences of the deadline.
fn set_fence_deadline(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut vbltime: i64 = 0;
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let mut v: i64 = 0;
if !drm_atomic_crtc_needs_modeset(new_crtc_state as *const drm_crtc_state) {
if unsafe { (*new_crtc_state).active } {
if (unsafe { drm_crtc_next_vblank_start(crtc, core::ptr::addr_of_mut!(v)) }) == 0 {
let __v3: bool = vbltime == 0 || ktime_before(v, vbltime);
if __v3 {
vbltime = v;
}
}
}
}
}
i = i.wrapping_add(1);
}
// If no CRTCs updated, then nothing to do:
if vbltime == 0 {
return;
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v6: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
let plane: *mut drm_plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v6 {
if !unsafe { (*new_plane_state).fence }.is_null() {
unsafe { dma_fence_set_deadline((*new_plane_state).fence, vbltime) };
}
}
i = i.wrapping_add(1);
}
}

/// drm_atomic_helper_wait_for_fences - wait for fences stashed in plane state
/// @dev: DRM device
/// @state: atomic state object with old state structures
/// @pre_swap: If true, do an interruptible wait, and @state is the new state.
/// 	Otherwise @state is the old state.
///
/// For implicit sync, driver should fish the exclusive fence out from the
/// incoming fb's and stash it in the drm_plane_state.  This is called after
/// drm_atomic_helper_swap_state() so it uses the current plane state (and
/// just uses the atomic state to find the changed planes)
///
/// Note that @pre_swap is needed since the point where we block for fences moves
/// around depending upon whether an atomic commit is blocking or
/// non-blocking. For non-blocking commit all waiting needs to happen after
/// drm_atomic_helper_swap_state() is called, but for blocking commits we want
/// to wait **before** we do anything that can't be easily rolled back. That is
/// before we call drm_atomic_helper_swap_state().
///
/// Returns zero if success or < 0 if dma_fence_wait() fails.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_wait_for_fences")]
extern "C" fn drm_atomic_helper_wait_for_fences(dev: *mut drm_device, state: *mut drm_atomic_commit, pre_swap: bool) -> i32 {
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i32 = 0;
set_fence_deadline(dev, state);
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v0: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
let plane: *mut drm_plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if !unsafe { (*new_plane_state).fence }.is_null() {
let mut __v3: i64;
let __ret_warn_on: i32 = unsafe { (*new_plane_state).fb }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("614: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 614b - ., 3; .popsection\n# {0}", const 614 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_wait_for_fences_4, sym __slate_asm_string_drm_atomic_helper_wait_for_fences_5, const 1854 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("615: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 615b - ., 4; .popsection\n# {0}", const 615 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
// If waiting for fences pre-swap (ie: nonblock), userspace can
// still interrupt the operation. Instead of blocking until the
// timer expires, make the wait interruptible.
ret = dma_fence_wait(unsafe { (*new_plane_state).fence }, pre_swap) as i32;
if ret != 0 {
return ret;
}
dma_fence_put(unsafe { (*new_plane_state).fence });
unsafe {
(*new_plane_state).fence = core::ptr::null_mut::<dma_fence>();
}
}
}
i = i.wrapping_add(1);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_wait_for_fences_616: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, bool) -> i32>, *mut ()>(Some(drm_atomic_helper_wait_for_fences)) };

/// drm_atomic_helper_wait_for_vblanks - wait for vblank on CRTCs
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// Helper to, after atomic commit, wait for vblanks on all affected
/// CRTCs (ie. before cleaning up old framebuffers using
/// drm_atomic_helper_cleanup_planes()). It will only wait on CRTCs where the
/// framebuffers have actually changed to optimize for the legacy cursor and
/// plane update use-case.
///
/// Drivers using the nonblocking commit tracking support initialized by calling
/// drm_atomic_helper_setup_commit() should look at
/// drm_atomic_helper_wait_for_flip_done() as an alternative.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_wait_for_vblanks")]
extern "C" fn drm_atomic_helper_wait_for_vblanks(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v4: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v4) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __cond: bool = unsafe { core::mem::zeroed() };
let mut __cond_1: bool = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i64 = unsafe { core::mem::zeroed() };
let mut __v20: bool = unsafe { core::mem::zeroed() };
let mut __v28: bool = unsafe { core::mem::zeroed() };
let mut crtc_mask: u32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut new_crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(new_crtc_state) as *mut *mut drm_crtc_state;
let mut old_crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(old_crtc_state) as *mut *mut drm_crtc_state;
let mut crtc: core::mem::MaybeUninit<*mut drm_crtc> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut *mut drm_crtc = core::ptr::addr_of_mut!(crtc) as *mut *mut drm_crtc;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
crtc_mask = 0;
// Legacy cursor ioctls are completely unsynced, and userspace
// relies on that (by doing tons of cursor updates).
if ((*state).__slate_bits_0.__get_legacy_cursor_update() as i32) != 0 {
return;
} else {
*__slate_slot_31 = 0;
while *__slate_slot_31 < (*(*state).dev).mode_config.num_crtc {
if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).ptr }.is_null() {
*__slate_slot_34 = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).ptr;
*__slate_slot_34;
*__slate_slot_33 = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).old_state;
*__slate_slot_33;
*__slate_slot_32 = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).new_state;
*__slate_slot_32;
__v28 = true;
} else {
__v28 = false;
}
if __v28 {
if (*(*__slate_slot_32)).active {
*__slate_slot_30 = drm_crtc_vblank_get(*__slate_slot_34);
if *__slate_slot_30 == 0 {
crtc_mask |= drm_crtc_mask(*__slate_slot_34 as *const drm_crtc);
(*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).last_vblank_count = drm_crtc_vblank_count(*__slate_slot_34);
}
}
}
let __v22: i32 = *__slate_slot_31;
*__slate_slot_31 = __v22.wrapping_add(1);
}
*__slate_slot_31 = 0;
while *__slate_slot_31 < (*(*state).dev).mode_config.num_crtc {
if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).ptr }.is_null() {
*__slate_slot_34 = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).ptr;
*__slate_slot_34;
*__slate_slot_33 = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).old_state;
__v20 = true;
} else {
__v20 = false;
}
if __v20 {
let queue: *mut wait_queue_head = drm_crtc_vblank_waitqueue(*__slate_slot_34);
if crtc_mask & drm_crtc_mask(*__slate_slot_34 as *const drm_crtc) != 0 {
__ret_1 = msecs_to_jiffies(1000u32) as i64;
might_sleep!();
if !___wait_cond_timeout!((*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).last_vblank_count != drm_crtc_vblank_count(*__slate_slot_34), __ret_1) {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = msecs_to_jiffies(1000u32) as i64;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_14), if false { 1i32 } else { 0i32 });
loop {
prepare_to_wait_event(queue, core::ptr::addr_of_mut!(*__slate_slot_14), 2i32);
__cond_1 = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).last_vblank_count != drm_crtc_vblank_count(*__slate_slot_34);
if __cond_1 && __ret == 0 {
__ret = 1;
}
let __v11: bool = __cond_1 || __ret == 0;
if __v11 {
break;
} else {
__ret = schedule_timeout(__ret);
__cond = (*unsafe { (*state).crtcs }.wrapping_offset(*__slate_slot_31 as isize)).last_vblank_count != drm_crtc_vblank_count(*__slate_slot_34);
if __cond && __ret == 0 {
__ret = 1;
}
if __cond || __ret == 0 {
break;
}
}
}
finish_wait(queue, core::ptr::addr_of_mut!(*__slate_slot_14));
__ret_1 = __ret;
}
*__slate_slot_30 = __ret_1 as i32;
let __ret_warn_on: i32 = (*__slate_slot_30 == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"[CRTC:%d:%s] vblank wait timed out\n".as_ptr() as *const u8, (*(*__slate_slot_34)).base.id, (*(*__slate_slot_34)).name);
*__slate_slot_4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_drm_atomic_helper_wait_for_vblanks_35, sym __slate_asm_string_drm_atomic_helper_wait_for_vblanks_36, const 1928 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_2 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_4).unwrap()(*__slate_slot_2, (*(*__slate_slot_34)).base.id, (*(*__slate_slot_34)).name);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
drm_crtc_vblank_put(*__slate_slot_34);
}
}
let __v1: i32 = *__slate_slot_31;
*__slate_slot_31 = __v1.wrapping_add(1);
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_617: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_wait_for_vblanks_618: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_wait_for_vblanks)) };

/// drm_atomic_helper_wait_for_flip_done - wait for all page flips to be done
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// Helper to, after atomic commit, wait for page flips on all affected
/// crtcs (ie. before cleaning up old framebuffers using
/// drm_atomic_helper_cleanup_planes()). Compared to
/// drm_atomic_helper_wait_for_vblanks() this waits for the completion on all
/// CRTCs, assuming that cursors-only updates are signalling their completion
/// immediately (or using a different path).
///
/// This requires that drivers use the nonblocking commit tracking support
/// initialized using drm_atomic_helper_setup_commit().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_wait_for_flip_done")]
extern "C" fn drm_atomic_helper_wait_for_flip_done(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut i: i32 = 0;
while i < unsafe { (*dev).mode_config.num_crtc } {
let commit: *mut drm_crtc_commit = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).commit };
let mut ret: i32 = 0;
let crtc: *mut drm_crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
if !(crtc.is_null() || commit.is_null()) {
ret = (unsafe { wait_for_completion_timeout(core::ptr::addr_of_mut!((*commit).flip_done), 10000u64) }) as i32;
if ret == 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* [CRTC:%d:%s] flip_done timed out\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
}
}
i = i.wrapping_add(1);
}
if !unsafe { (*state).fake_commit }.is_null() {
unsafe { complete_all(core::ptr::addr_of_mut!((*(*state).fake_commit).flip_done)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_wait_for_flip_done_619: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_wait_for_flip_done)) };

/// drm_atomic_helper_commit_tail - commit atomic update to hardware
/// @state: atomic state object being committed
///
/// This is the default implementation for the
/// &drm_mode_config_helper_funcs.atomic_commit_tail hook, for drivers
/// that do not support runtime_pm or do not need the CRTC to be
/// enabled to perform a commit. Otherwise, see
/// drm_atomic_helper_commit_tail_rpm().
///
/// Note that the default ordering of how the various stages are called is to
/// match the legacy modeset helper library closest.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_tail")]
extern "C" fn drm_atomic_helper_commit_tail(state: *mut drm_atomic_commit) {
let dev: *mut drm_device = unsafe { (*state).dev };
drm_atomic_helper_commit_modeset_disables(dev, state);
drm_atomic_helper_commit_planes(dev, state, 0u32);
drm_atomic_helper_commit_modeset_enables(dev, state);
drm_atomic_helper_fake_vblank(state);
drm_atomic_helper_commit_hw_done(state);
drm_atomic_helper_wait_for_vblanks(dev, state);
drm_atomic_helper_cleanup_planes(dev, state);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_tail_620: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_tail)) };

/// drm_atomic_helper_commit_tail_rpm - commit atomic update to hardware
/// @state: new modeset state to be committed
///
/// This is an alternative implementation for the
/// &drm_mode_config_helper_funcs.atomic_commit_tail hook, for drivers
/// that support runtime_pm or need the CRTC to be enabled to perform a
/// commit. Otherwise, one should use the default implementation
/// drm_atomic_helper_commit_tail().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_tail_rpm")]
extern "C" fn drm_atomic_helper_commit_tail_rpm(state: *mut drm_atomic_commit) {
let dev: *mut drm_device = unsafe { (*state).dev };
drm_atomic_helper_commit_modeset_disables(dev, state);
drm_atomic_helper_commit_modeset_enables(dev, state);
drm_atomic_helper_commit_planes(dev, state, BIT!(0i32) as u32);
drm_atomic_helper_fake_vblank(state);
drm_atomic_helper_commit_hw_done(state);
drm_atomic_helper_wait_for_vblanks(dev, state);
drm_atomic_helper_cleanup_planes(dev, state);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_tail_rpm_621: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_tail_rpm)) };

fn commit_tail(state: *mut drm_atomic_commit) {
let dev: *mut drm_device = unsafe { (*state).dev };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut start: i64 = 0;
let mut commit_time_ms: i64 = 0;
// We cannot safely access new_crtc_state after
// drm_atomic_helper_commit_hw_done() so figure out which crtc's have
// self-refresh active beforehand:
let mut i: u32 = 0;
let mut new_self_refresh_mask: u32 = 0;
let funcs: *const drm_mode_config_helper_funcs = unsafe { (*dev).mode_config.helper_private };
// We're measuring the _entire_ commit, so the time will vary depending
// on how many fences and objects are involved. For the purposes of self
// refresh, this is desirable since it'll give us an idea of how
// congested things are. This will inform our decision on how often we
// should enter self refresh after idle.
//
// These times will be averaged out in the self refresh helpers to avoid
// overreacting over one outlier frame
start = unsafe { ktime_get() };
drm_atomic_helper_wait_for_fences(dev, state, false);
drm_atomic_helper_wait_for_dependencies(state);
while i < ((unsafe { (*(*state).dev).mode_config.num_crtc }) as u32) {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
let crtc: *mut drm_crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if unsafe { (*new_crtc_state).self_refresh_active } {
let __v2: u32 = new_self_refresh_mask;
new_self_refresh_mask = ((__v2 as u64) | BIT!(i)) as u32;
}
}
i = i.wrapping_add(1);
}
if !funcs.is_null() && (unsafe { (*funcs).atomic_commit_tail }) != None {
unsafe { unsafe { (*funcs).atomic_commit_tail }.unwrap()(state) };
} else {
drm_atomic_helper_commit_tail(state);
}
commit_time_ms = ktime_ms_delta(unsafe { ktime_get() }, start);
if commit_time_ms > 0 {
unsafe { drm_self_refresh_helper_update_avg_times(state, (commit_time_ms as u64) as u32, new_self_refresh_mask) };
}
drm_atomic_helper_commit_cleanup_done(state);
drm_atomic_commit_put(state);
}

#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.commit_work")]
extern "C" fn commit_work(work: *mut work_struct) {
let state: *mut drm_atomic_commit = container_of__50!(work, drm_atomic_commit);
commit_tail(state);
}

/// drm_atomic_helper_async_check - check if state can be committed asynchronously
/// @dev: DRM device
/// @state: the driver state object
///
/// This helper will check if it is possible to commit the state asynchronously.
/// Async commits are not supposed to swap the states like normal sync commits
/// but just do in-place changes on the current state.
///
/// It will return 0 if the commit can happen in an asynchronous fashion or error
/// if not. Note that error just mean it can't be committed asynchronously, if it
/// fails the commit should be treated like a normal synchronous commit.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_async_check")]
extern "C" fn drm_atomic_helper_async_check(dev: *mut drm_device, state: *mut drm_atomic_commit) -> i32 {
let mut crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut plane: *mut drm_plane = core::ptr::null_mut::<drm_plane>();
let mut old_plane_state: *mut drm_plane_state = core::ptr::null_mut::<drm_plane_state>();
let mut new_plane_state: *mut drm_plane_state = core::ptr::null_mut::<drm_plane_state>();
let mut i: i32 = 0;
let mut ret: i32 = 0;
let mut n_planes: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
let crtc: *mut drm_crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
if drm_atomic_crtc_needs_modeset(crtc_state as *const drm_crtc_state) {
return 22i32.wrapping_neg();
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v5: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v5 {
n_planes = n_planes.wrapping_add(1);
}
i = i.wrapping_add(1);
}
// FIXME: we support only single plane updates for now
if n_planes != 1 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"only single plane async updates are supported\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
if unsafe { (*new_plane_state).crtc }.is_null() || (unsafe { (*old_plane_state).crtc }) != unsafe { (*new_plane_state).crtc } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] async update cannot change CRTC\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
return 22i32.wrapping_neg();
}
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if (unsafe { (*funcs).atomic_async_update }) == None {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] driver does not support async updates\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
return 22i32.wrapping_neg();
}
if !unsafe { (*new_plane_state).fence }.is_null() {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] missing fence for async update\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
return 22i32.wrapping_neg();
}
// Don't do an async update if there is an outstanding commit modifying
// the plane.  This prevents our async update's changes from getting
// overridden by a previous synchronous update's state.
let __v13: bool = !unsafe { (*old_plane_state).commit }.is_null() && !unsafe { try_wait_for_completion(core::ptr::addr_of_mut!((*(*old_plane_state).commit).hw_done)) };
if __v13 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] inflight previous commit preventing async commit\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
return 16i32.wrapping_neg();
}
ret = unsafe { unsafe { (*funcs).atomic_async_check }.unwrap()(plane, state, false) };
if ret != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev(dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] driver async check failed\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_async_check_622: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_async_check)) };

/// drm_atomic_helper_async_commit - commit state asynchronously
/// @dev: DRM device
/// @state: the driver state object
///
/// This function commits a state asynchronously, i.e., not vblank
/// synchronized. It should be used on a state only when
/// drm_atomic_async_check() succeeds. Async commits are not supposed to swap
/// the states like normal sync commits, but just do in-place changes on the
/// current state.
///
/// TODO: Implement full swap instead of doing in-place changes.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_async_commit")]
extern "C" fn drm_atomic_helper_async_commit(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v0: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let new_fb: *mut drm_framebuffer = unsafe { (*plane_state).fb };
let old_fb: *mut drm_framebuffer = unsafe { (*(*plane).state).fb };
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
unsafe { unsafe { (*funcs).atomic_async_update }.unwrap()(plane, state) };
// ->atomic_async_update() is supposed to update the
// plane->state in-place, make sure at least common
// properties have been properly updated.
let mut __v2: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*(*plane).state).fb }) != new_fb) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("623: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 623b - ., 3; .popsection\n# {0}", const 623 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_async_commit_3, sym __slate_asm_string_drm_atomic_helper_async_commit_4, const 2218 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("624: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 624b - ., 4; .popsection\n# {0}", const 624 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
let mut __v7: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*(*plane).state).crtc_x }) != unsafe { (*plane_state).crtc_x }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("625: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 625b - ., 3; .popsection\n# {0}", const 625 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_async_commit_8, sym __slate_asm_string_drm_atomic_helper_async_commit_9, const 2219 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("626: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 626b - ., 4; .popsection\n# {0}", const 626 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
}
let mut __v12: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*(*plane).state).crtc_y }) != unsafe { (*plane_state).crtc_y }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("627: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 627b - ., 3; .popsection\n# {0}", const 627 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_async_commit_13, sym __slate_asm_string_drm_atomic_helper_async_commit_14, const 2220 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("628: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 628b - ., 4; .popsection\n# {0}", const 628 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v12 = (__ret_warn_on != 0) as i64;
}
let mut __v17: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*(*plane).state).src_x }) != unsafe { (*plane_state).src_x }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("629: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 629b - ., 3; .popsection\n# {0}", const 629 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_async_commit_18, sym __slate_asm_string_drm_atomic_helper_async_commit_19, const 2221 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("630: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 630b - ., 4; .popsection\n# {0}", const 630 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v17 = (__ret_warn_on != 0) as i64;
}
let mut __v22: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*(*plane).state).src_y }) != unsafe { (*plane_state).src_y }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("631: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 631b - ., 3; .popsection\n# {0}", const 631 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_async_commit_23, sym __slate_asm_string_drm_atomic_helper_async_commit_24, const 2222 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("632: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 632b - ., 4; .popsection\n# {0}", const 632 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v22 = (__ret_warn_on != 0) as i64;
}
// Make sure the FBs have been swapped so that cleanups in the
// new_state performs a cleanup in the old FB.
let mut __v27: i64;
let __ret_warn_on: i32 = ((unsafe { (*plane_state).fb }) != old_fb) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("633: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 633b - ., 3; .popsection\n# {0}", const 633 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_async_commit_28, sym __slate_asm_string_drm_atomic_helper_async_commit_29, const 2228 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("634: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 634b - ., 4; .popsection\n# {0}", const 634 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v27 = (__ret_warn_on != 0) as i64;
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_async_commit_635: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_async_commit)) };

/// drm_atomic_helper_commit - commit validated state object
/// @dev: DRM device
/// @state: the driver state object
/// @nonblock: whether nonblocking behavior is requested.
///
/// This function commits a with drm_atomic_helper_check() pre-validated state
/// object. This can still fail when e.g. the framebuffer reservation fails. This
/// function implements nonblocking commits, using
/// drm_atomic_helper_setup_commit() and related functions.
///
/// Committing the actual hardware state is done through the
/// &drm_mode_config_helper_funcs.atomic_commit_tail callback, or its default
/// implementation drm_atomic_helper_commit_tail().
///
/// RETURNS:
/// Zero for success or -errno.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit")]
extern "C" fn drm_atomic_helper_commit(dev: *mut drm_device, state: *mut drm_atomic_commit, nonblock: bool) -> i32 {
let mut __slate_compound_1: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ((*state).__slate_bits_0.__get_async_update() as i32) != 0 {
*__slate_slot_0 = drm_atomic_helper_prepare_planes(dev, state);
if *__slate_slot_0 != 0 {
return *__slate_slot_0;
} else {
drm_atomic_helper_async_commit(dev, state);
drm_atomic_helper_unprepare_planes(dev, state);
return 0;
}
} else {
*__slate_slot_0 = drm_atomic_helper_setup_commit(state, nonblock);
if *__slate_slot_0 != 0 {
return *__slate_slot_0;
} else {
loop {
__init_work(core::ptr::addr_of_mut!((*state).commit_work), 0i32);
(*core::ptr::addr_of_mut!((*state).commit_work)).data = *{
__slate_compound_1 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_1)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*state).commit_work)).entry));
(*core::ptr::addr_of_mut!((*state).commit_work)).func = Some(commit_work);
if true {
if true {
break;
}
}
}
*__slate_slot_0 = drm_atomic_helper_prepare_planes(dev, state);
if *__slate_slot_0 != 0 {
return *__slate_slot_0;
} else {
'__join_0: {
if !nonblock {
*__slate_slot_0 = drm_atomic_helper_wait_for_fences(dev, state, true);
if *__slate_slot_0 != 0 {
break '__join_0;
}
}
// This is the point of no return - everything below never fails except
// when the hw goes bonghits. Which means we can commit the new state on
// the software side now.
*__slate_slot_0 = drm_atomic_helper_swap_state(state, true);
if *__slate_slot_0 == 0 {
// Everything below can be run asynchronously without the need to grab
// any modeset locks at all under one condition: It must be guaranteed
// that the asynchronous work has either been cancelled (if the driver
// supports it, which at least requires that the framebuffers get
// cleaned up with drm_atomic_helper_cleanup_planes()) or completed
// before the new state gets committed on the software side with
// drm_atomic_helper_swap_state().
//
// This scheme allows new atomic state updates to be prepared and
// checked in parallel to the asynchronous completion of the previous
// update. Which is important since compositors need to figure out the
// composition of the next frame right after having submitted the
// current layout.
//
// NOTE: Commit work has multiple phases, first hardware commit, then
// cleanup. We want them to overlap, hence need system_dfl_wq to
// make sure work items don't artificially stall on each another.
drm_atomic_commit_get(state);
if nonblock {
queue_work(system_dfl_wq, core::ptr::addr_of_mut!((*state).commit_work));
} else {
commit_tail(state);
}
return 0;
}
}
drm_atomic_helper_unprepare_planes(dev, state);
return *__slate_slot_0;
}
}
}
}
}

static mut __key__2272: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_636: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, bool) -> i32>, *mut ()>(Some(drm_atomic_helper_commit)) };

// DOC: implementing nonblocking commit
//
// Nonblocking atomic commits should use struct &drm_crtc_commit to sequence
// different operations against each another. Locks, especially struct
// &drm_modeset_lock, should not be held in worker threads or any other
// asynchronous context used to commit the hardware state.
//
// drm_atomic_helper_commit() implements the recommended sequence for
// nonblocking commits, using drm_atomic_helper_setup_commit() internally:
//
// 1. Run drm_atomic_helper_prepare_planes(). Since this can fail and we
// need to propagate out of memory/VRAM errors to userspace, it must be called
// synchronously.
//
// 2. Synchronize with any outstanding nonblocking commit worker threads which
// might be affected by the new state update. This is handled by
// drm_atomic_helper_setup_commit().
//
// Asynchronous workers need to have sufficient parallelism to be able to run
// different atomic commits on different CRTCs in parallel. The simplest way to
// achieve this is by running them on the &system_dfl_wq work queue. Note
// that drivers are not required to split up atomic commits and run an
// individual commit in parallel - userspace is supposed to do that if it cares.
// But it might be beneficial to do that for modesets, since those necessarily
// must be done as one global operation, and enabling or disabling a CRTC can
// take a long time. But even that is not required.
//
// IMPORTANT: A &drm_atomic_commit update for multiple CRTCs is sequenced
// against all CRTCs therein. Therefore for atomic state updates which only flip
// planes the driver must not get the struct &drm_crtc_state of unrelated CRTCs
// in its atomic check code: This would prevent committing of atomic updates to
// multiple CRTCs in parallel. In general, adding additional state structures
// should be avoided as much as possible, because this reduces parallelism in
// (nonblocking) commits, both due to locking and due to commit sequencing
// requirements.
//
// 3. The software state is updated synchronously with
// drm_atomic_helper_swap_state(). Doing this under the protection of all modeset
// locks means concurrent callers never see inconsistent state. Note that commit
// workers do not hold any locks; their access is only coordinated through
// ordering. If workers would access state only through the pointers in the
// free-standing state objects (currently not the case for any driver) then even
// multiple pending commits could be in-flight at the same time.
//
// 4. Schedule a work item to do all subsequent steps, using the split-out
// commit helpers: a) pre-plane commit b) plane commit c) post-plane commit and
// then cleaning up the framebuffers after the old framebuffer is no longer
// being displayed. The scheduled work should synchronize against other workers
// using the &drm_crtc_commit infrastructure as needed. See
// drm_atomic_helper_setup_commit() for more details.
fn stall_checks(crtc: *mut drm_crtc, nonblock: bool) -> i32 {
let mut commit: *mut drm_crtc_commit = unsafe { core::mem::zeroed() };
let mut stall_commit: *mut drm_crtc_commit = core::ptr::null_mut::<drm_crtc_commit>();
let mut completed: bool = true;
let mut i: i32 = 0;
let mut ret: i64 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
commit = list_first_entry__23!(unsafe { core::ptr::addr_of_mut!((*crtc).commit_list) });
while list_entry_is_head!(commit, unsafe { core::ptr::addr_of_mut!((*crtc).commit_list) }, commit_entry) == 0 {
if i == 0 {
completed = unsafe { try_wait_for_completion(core::ptr::addr_of_mut!((*commit).flip_done)) };
// Userspace is not allowed to get ahead of the previous
// commit with nonblocking ones.
if !completed && nonblock {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*crtc).dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] busy with a previous commit\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
return 16i32.wrapping_neg();
}
} else {
if i == 1 {
stall_commit = drm_crtc_commit_get(commit);
break;
}
}
i = i.wrapping_add(1);
commit = list_next_entry__30!(commit, commit_entry);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
if stall_commit.is_null() {
return 0;
}
// We don't want to let commits get ahead of cleanup work too much,
// stalling on 2nd previous commit means triple-buffer won't ever stall.
ret = unsafe { wait_for_completion_interruptible_timeout(core::ptr::addr_of_mut!((*stall_commit).cleanup_done), 10000u64) };
if ret == 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*crtc).dev as *const drm_device) as *const device, c"[drm] *ERROR* [CRTC:%d:%s] cleanup_done timed out\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
}
drm_crtc_commit_put(stall_commit);
(if ret < 0 { ret } else { 0i64 }) as i32
}

#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.release_crtc_commit")]
extern "C" fn release_crtc_commit(completion: *mut completion) {
let commit: *mut drm_crtc_commit = container_of__57!(completion);
drm_crtc_commit_put(commit);
}

fn init_commit(commit: *mut drm_crtc_commit, crtc: *mut drm_crtc) {
init_completion(unsafe { core::ptr::addr_of_mut!((*commit).flip_done) });
init_completion(unsafe { core::ptr::addr_of_mut!((*commit).hw_done) });
init_completion(unsafe { core::ptr::addr_of_mut!((*commit).cleanup_done) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*commit).commit_entry) });
kref_init(unsafe { core::ptr::addr_of_mut!((*commit).r#ref) });
unsafe {
(*commit).crtc = crtc;
}
}

fn crtc_or_fake_commit(state: *mut drm_atomic_commit, crtc: *mut drm_crtc) -> *mut drm_crtc_commit {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
if !crtc.is_null() {
let new_crtc_state: *mut drm_crtc_state = drm_atomic_get_new_crtc_state(state as *const drm_atomic_commit, crtc);
return unsafe { (*new_crtc_state).commit };
}
if unsafe { (*state).fake_commit }.is_null() {
let mut __v0: *mut drm_crtc_commit;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_crtc_commit>() as u64, 1u64);
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
__v0 = __v1 as *mut drm_crtc_commit;
unsafe {
(*state).fake_commit = __v0;
}
if unsafe { (*state).fake_commit }.is_null() {
return core::ptr::null_mut::<drm_crtc_commit>();
}
init_commit(unsafe { (*state).fake_commit }, core::ptr::null_mut::<drm_crtc>());
}
unsafe { (*state).fake_commit }
}

/// drm_atomic_helper_setup_commit - setup possibly nonblocking commit
/// @state: new modeset state to be committed
/// @nonblock: whether nonblocking behavior is requested.
///
/// This function prepares @state to be used by the atomic helper's support for
/// nonblocking commits. Drivers using the nonblocking commit infrastructure
/// should always call this function from their
/// &drm_mode_config_funcs.atomic_commit hook.
///
/// Drivers that need to extend the commit setup to private objects can use the
/// &drm_mode_config_helper_funcs.atomic_commit_setup hook.
///
/// To be able to use this support drivers need to use a few more helper
/// functions. drm_atomic_helper_wait_for_dependencies() must be called before
/// actually committing the hardware state, and for nonblocking commits this call
/// must be placed in the async worker. See also drm_atomic_helper_swap_state()
/// and its stall parameter, for when a driver's commit hooks look at the
/// &drm_crtc.state, &drm_plane.state or &drm_connector.state pointer directly.
///
/// Completion of the hardware commit step must be signalled using
/// drm_atomic_helper_commit_hw_done(). After this step the driver is not allowed
/// to read or change any permanent software or hardware modeset state. The only
/// exception is state protected by other means than &drm_modeset_lock locks.
/// Only the free standing @state with pointers to the old state structures can
/// be inspected, e.g. to clean up old buffers using
/// drm_atomic_helper_cleanup_planes().
///
/// At the very end, before cleaning up @state drivers must call
/// drm_atomic_helper_commit_cleanup_done().
///
/// This is all implemented by in drm_atomic_helper_commit(), giving drivers a
/// complete and easy-to-use default implementation of the atomic_commit() hook.
///
/// The tracking of asynchronously executed and still pending commits is done
/// using the core structure &drm_crtc_commit.
///
/// By default there's no need to clean up resources allocated by this function
/// explicitly: drm_atomic_commit_default_clear() will take care of that
/// automatically.
///
/// Returns:
/// 0 on success. -EBUSY when userspace schedules nonblocking commits too fast,
/// -ENOMEM on allocation failures and -EINTR when a signal is pending.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_setup_commit")]
extern "C" fn drm_atomic_helper_setup_commit(state: *mut drm_atomic_commit, nonblock: bool) -> i32 {
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut conn: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut old_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut commit: *mut drm_crtc_commit = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i32 = 0;
let funcs: *const drm_mode_config_helper_funcs = unsafe { (*(*state).dev).mode_config.helper_private };
'__slate_break_9: while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let mut __v3: *mut drm_crtc_commit;
{
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_crtc_commit>() as u64, 1u64);
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
__v3 = __v4 as *mut drm_crtc_commit;
}
commit = __v3;
if commit.is_null() {
return 12i32.wrapping_neg();
}
init_commit(commit, crtc);
unsafe {
(*new_crtc_state).commit = commit;
}
ret = stall_checks(crtc, nonblock);
if ret != 0 {
return ret;
}
// Drivers only send out events when at least either current or
// new CRTC state is active. Complete right away if everything
// stays off.
if !unsafe { (*old_crtc_state).active } && !unsafe { (*new_crtc_state).active } {
unsafe { complete_all(core::ptr::addr_of_mut!((*commit).flip_done)) };
} else {
// Legacy cursor updates are fully unsynced.
if ((unsafe { (*state).__slate_bits_0.__get_legacy_cursor_update() }) as i32) != 0 {
unsafe { complete_all(core::ptr::addr_of_mut!((*commit).flip_done)) };
} else {
if unsafe { (*new_crtc_state).event }.is_null() {
let mut __v10: *mut drm_pending_vblank_event;
let __obj_size: u64 = size_mul(core::mem::size_of::<drm_pending_vblank_event>() as u64, 1u64);
let mut __v11: *mut ();
let mut __v12: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
} });
}
__v12 = _res;
__v11 = __v12;
__v10 = __v11 as *mut drm_pending_vblank_event;
unsafe {
(*commit).event = __v10;
}
if unsafe { (*commit).event }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*new_crtc_state).event = (*commit).event;
}
}
unsafe {
(*(*new_crtc_state).event).base.completion = core::ptr::addr_of_mut!((*commit).flip_done);
}
unsafe {
(*(*new_crtc_state).event).base.completion_release = Some(release_crtc_commit);
}
drm_crtc_commit_get(commit);
unsafe {
(*commit).abort_completion = true;
}
unsafe {
(*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).commit = commit;
}
drm_crtc_commit_get(commit);
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v18: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
conn = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v18 {
// Userspace is not allowed to get ahead of the previous
// commit with nonblocking ones.
let __v21: bool = nonblock && !unsafe { (*old_conn_state).commit }.is_null() && !unsafe { try_wait_for_completion(core::ptr::addr_of_mut!((*(*old_conn_state).commit).flip_done)) };
if __v21 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*conn).dev as *const drm_device) as *const device, 4u32, c"[CONNECTOR:%d:%s] busy with a previous commit\n".as_ptr() as *const u8, (*conn).base.id, (*conn).name) };
return 16i32.wrapping_neg();
}
// Always track connectors explicitly for e.g. link retraining.
let __v22: *mut drm_crtc = unsafe { (*new_conn_state).crtc };
commit = crtc_or_fake_commit(state, if !__v22.is_null() { __v22 } else { unsafe { (*old_conn_state).crtc } });
if commit.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*new_conn_state).commit = drm_crtc_commit_get(commit);
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v26: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v26 {
// Userspace is not allowed to get ahead of the previous
// commit with nonblocking ones.
let __v29: bool = nonblock && !unsafe { (*old_plane_state).commit }.is_null() && !unsafe { try_wait_for_completion(core::ptr::addr_of_mut!((*(*old_plane_state).commit).flip_done)) };
if __v29 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*plane).dev as *const drm_device) as *const device, 4u32, c"[PLANE:%d:%s] busy with a previous commit\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
return 16i32.wrapping_neg();
}
// Always track planes explicitly for async pageflip support.
let __v30: *mut drm_crtc = unsafe { (*new_plane_state).crtc };
commit = crtc_or_fake_commit(state, if !__v30.is_null() { __v30 } else { unsafe { (*old_plane_state).crtc } });
if commit.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*new_plane_state).commit = drm_crtc_commit_get(commit);
}
}
i = i.wrapping_add(1);
}
if !funcs.is_null() && (unsafe { (*funcs).atomic_commit_setup }) != None {
return unsafe { unsafe { (*funcs).atomic_commit_setup }.unwrap()(state) };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_setup_commit_637: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, bool) -> i32>, *mut ()>(Some(drm_atomic_helper_setup_commit)) };

/// drm_atomic_helper_wait_for_dependencies - wait for required preceding commits
/// @state: atomic state object being committed
///
/// This function waits for all preceding commits that touch the same CRTC as
/// @state to both be committed to the hardware (as signalled by
/// drm_atomic_helper_commit_hw_done()) and executed by the hardware (as signalled
/// by calling drm_crtc_send_vblank_event() on the &drm_crtc_state.event).
///
/// This is part of the atomic helper support for nonblocking commits, see
/// drm_atomic_helper_setup_commit() for an overview.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_wait_for_dependencies")]
extern "C" fn drm_atomic_helper_wait_for_dependencies(state: *mut drm_atomic_commit) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut old_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut conn: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut ret: i64 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v0 {
ret = (unsafe { drm_crtc_commit_wait((*old_crtc_state).commit) }) as i64;
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*crtc).dev as *const drm_device) as *const device, c"[drm] *ERROR* [CRTC:%d:%s] commit wait timed out\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v6: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
conn = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v6 {
ret = (unsafe { drm_crtc_commit_wait((*old_conn_state).commit) }) as i64;
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*conn).dev as *const drm_device) as *const device, c"[drm] *ERROR* [CONNECTOR:%d:%s] commit wait timed out\n".as_ptr() as *const u8, (*conn).base.id, (*conn).name) };
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v12: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v12 {
ret = (unsafe { drm_crtc_commit_wait((*old_plane_state).commit) }) as i64;
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*plane).dev as *const drm_device) as *const device, c"[drm] *ERROR* [PLANE:%d:%s] commit wait timed out\n".as_ptr() as *const u8, (*plane).base.id, (*plane).name) };
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_wait_for_dependencies_638: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_wait_for_dependencies)) };

/// drm_atomic_helper_fake_vblank - fake VBLANK events if needed
/// @state: atomic state object being committed
///
/// This function walks all CRTCs and fakes VBLANK events on those with
/// &drm_crtc_state.no_vblank set to true and &drm_crtc_state.event != NULL.
/// The primary use of this function is writeback connectors working in oneshot
/// mode and faking VBLANK events. In this case they only fake the VBLANK event
/// when a job is queued, and any change to the pipeline that does not touch the
/// connector is leading to timeouts when calling
/// drm_atomic_helper_wait_for_vblanks() or
/// drm_atomic_helper_wait_for_flip_done(). In addition to writeback
/// connectors, this function can also fake VBLANK events for CRTCs without
/// VBLANK interrupt.
///
/// This is part of the atomic helper support for nonblocking commits, see
/// drm_atomic_helper_setup_commit() for an overview.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_fake_vblank")]
extern "C" fn drm_atomic_helper_fake_vblank(state: *mut drm_atomic_commit) {
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let mut flags: u64 = 0;
if unsafe { (*new_crtc_state).no_vblank } {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*(*state).dev).event_lock) }, flags);
if !unsafe { (*new_crtc_state).event }.is_null() {
unsafe { drm_crtc_send_vblank_event(crtc, (*new_crtc_state).event) };
unsafe {
(*new_crtc_state).event = core::ptr::null_mut::<drm_pending_vblank_event>();
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*(*state).dev).event_lock) }, flags);
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_fake_vblank_639: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_fake_vblank)) };

/// drm_atomic_helper_commit_hw_done - setup possible nonblocking commit
/// @state: atomic state object being committed
///
/// This function is used to signal completion of the hardware commit step. After
/// this step the driver is not allowed to read or change any permanent software
/// or hardware modeset state. The only exception is state protected by other
/// means than &drm_modeset_lock locks.
///
/// Drivers should try to postpone any expensive or delayed cleanup work after
/// this function is called.
///
/// This is part of the atomic helper support for nonblocking commits, see
/// drm_atomic_helper_setup_commit() for an overview.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_hw_done")]
extern "C" fn drm_atomic_helper_commit_hw_done(state: *mut drm_atomic_commit) {
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut commit: *mut drm_crtc_commit = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
let crtc: *mut drm_crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
commit = unsafe { (*new_crtc_state).commit };
if !commit.is_null() {
// copy new_crtc_state->commit to old_crtc_state->commit,
// it's unsafe to touch new_crtc_state after hw_done,
// but we still need to do so in cleanup_done().
if !unsafe { (*old_crtc_state).commit }.is_null() {
drm_crtc_commit_put(unsafe { (*old_crtc_state).commit });
}
unsafe {
(*old_crtc_state).commit = drm_crtc_commit_get(commit);
}
// backend must have consumed any event by now
let mut __v4: i64;
let __ret_warn_on: i32 = !unsafe { (*new_crtc_state).event }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("640: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 640b - ., 3; .popsection\n# {0}", const 640 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_hw_done_5, sym __slate_asm_string_drm_atomic_helper_commit_hw_done_6, const 2758 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("641: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 641b - ., 4; .popsection\n# {0}", const 641 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
unsafe { complete_all(core::ptr::addr_of_mut!((*commit).hw_done)) };
}
}
i = i.wrapping_add(1);
}
if !unsafe { (*state).fake_commit }.is_null() {
unsafe { complete_all(core::ptr::addr_of_mut!((*(*state).fake_commit).hw_done)) };
unsafe { complete_all(core::ptr::addr_of_mut!((*(*state).fake_commit).flip_done)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_hw_done_642: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_hw_done)) };

/// drm_atomic_helper_commit_cleanup_done - signal completion of commit
/// @state: atomic state object being committed
///
/// This signals completion of the atomic update @state, including any
/// cleanup work. If used, it must be called right before calling
/// drm_atomic_commit_put().
///
/// This is part of the atomic helper support for nonblocking commits, see
/// drm_atomic_helper_setup_commit() for an overview.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_cleanup_done")]
extern "C" fn drm_atomic_helper_commit_cleanup_done(state: *mut drm_atomic_commit) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut commit: *mut drm_crtc_commit = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v0 {
commit = unsafe { (*old_crtc_state).commit };
let mut __v2: i64;
{
let __ret_warn_on: i32 = commit.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("643: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 643b - ., 3; .popsection\n# {0}", const 643 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_cleanup_done_3, sym __slate_asm_string_drm_atomic_helper_commit_cleanup_done_4, const 2789 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("644: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 644b - ., 4; .popsection\n# {0}", const 644 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
if __v2 == 0 {
unsafe { complete_all(core::ptr::addr_of_mut!((*commit).cleanup_done)) };
let mut __v8: i64;
let __ret_warn_on: i32 = !(!(!unsafe { try_wait_for_completion(core::ptr::addr_of_mut!((*commit).hw_done)) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("645: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 645b - ., 3; .popsection\n# {0}", const 645 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_cleanup_done_9, sym __slate_asm_string_drm_atomic_helper_commit_cleanup_done_10, const 2793 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("646: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 646b - ., 4; .popsection\n# {0}", const 646 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
spin_lock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
list_del(unsafe { core::ptr::addr_of_mut!((*commit).commit_entry) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
}
}
i = i.wrapping_add(1);
}
if !unsafe { (*state).fake_commit }.is_null() {
unsafe { complete_all(core::ptr::addr_of_mut!((*(*state).fake_commit).cleanup_done)) };
let mut __v15: i64;
let __ret_warn_on: i32 = !(!(!unsafe { try_wait_for_completion(core::ptr::addr_of_mut!((*(*state).fake_commit).hw_done)) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("647: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 647b - ., 3; .popsection\n# {0}", const 647 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_cleanup_done_16, sym __slate_asm_string_drm_atomic_helper_commit_cleanup_done_17, const 2802 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("648: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 648b - ., 4; .popsection\n# {0}", const 648 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v15 = (__ret_warn_on != 0) as i64;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_cleanup_done_649: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_commit_cleanup_done)) };

/// drm_atomic_helper_prepare_planes - prepare plane resources before commit
/// @dev: DRM device
/// @state: atomic state object with new state structures
///
/// This function prepares plane state, specifically framebuffers, for the new
/// configuration, by calling &drm_plane_helper_funcs.prepare_fb. If any failure
/// is encountered this function will call &drm_plane_helper_funcs.cleanup_fb on
/// any already successfully prepared framebuffer.
///
/// Returns:
/// 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_prepare_planes")]
extern "C" fn drm_atomic_helper_prepare_planes(dev: *mut drm_device, state: *mut drm_atomic_commit) -> i32 {
let mut funcs: core::mem::MaybeUninit<*const drm_plane_helper_funcs> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *const drm_plane_helper_funcs = core::ptr::addr_of_mut!(funcs) as *mut *const drm_plane_helper_funcs;
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut funcs_1: *const drm_plane_helper_funcs = unsafe { core::mem::zeroed() };
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut funcs_2: *const drm_plane_helper_funcs = unsafe { core::mem::zeroed() };
let mut __v14: bool = unsafe { core::mem::zeroed() };
let mut funcs: core::mem::MaybeUninit<*const drm_plane_helper_funcs> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *const drm_plane_helper_funcs = core::ptr::addr_of_mut!(funcs) as *mut *const drm_plane_helper_funcs;
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut __v26: bool = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut i32 = core::ptr::addr_of_mut!(j) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut new_plane_state: core::mem::MaybeUninit<*mut drm_plane_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut drm_plane_state = core::ptr::addr_of_mut!(new_plane_state) as *mut *mut drm_plane_state;
let mut plane: core::mem::MaybeUninit<*mut drm_plane> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut drm_plane = core::ptr::addr_of_mut!(plane) as *mut *mut drm_plane;
let mut new_conn_state: core::mem::MaybeUninit<*mut drm_connector_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut *mut drm_connector_state = core::ptr::addr_of_mut!(new_conn_state) as *mut *mut drm_connector_state;
let mut connector: core::mem::MaybeUninit<*mut drm_connector> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut drm_connector = core::ptr::addr_of_mut!(connector) as *mut *mut drm_connector;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
*__slate_slot_28 = 0;
while *__slate_slot_28 < (*state).num_connector {
if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(*__slate_slot_28 as isize)).ptr }.is_null() {
*__slate_slot_33 = (*unsafe { (*state).connectors }.wrapping_offset(*__slate_slot_28 as isize)).ptr;
*__slate_slot_33;
*__slate_slot_32 = (*unsafe { (*state).connectors }.wrapping_offset(*__slate_slot_28 as isize)).new_state;
*__slate_slot_32;
__v26 = true;
} else {
__v26 = false;
}
if __v26 {
if !unsafe { (*(*__slate_slot_32)).writeback_job }.is_null() {
*__slate_slot_29 = drm_writeback_prepare_job((*(*__slate_slot_32)).writeback_job);
if *__slate_slot_29 < 0 {
return *__slate_slot_29;
}
}
}
let __v24: i32 = *__slate_slot_28;
*__slate_slot_28 = __v24.wrapping_add(1);
}
*__slate_slot_28 = 0;
'__join_10: {
while *__slate_slot_28 < (*(*state).dev).mode_config.num_total_plane {
if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_28 as isize)).ptr }.is_null() {
*__slate_slot_31 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_28 as isize)).ptr;
*__slate_slot_31;
*__slate_slot_30 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_28 as isize)).new_state;
*__slate_slot_30;
__v22 = true;
} else {
__v22 = false;
}
if __v22 {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_20 = (*(*__slate_slot_31)).helper_private;
if (*(*__slate_slot_20)).prepare_fb != None {
*__slate_slot_29 = unsafe { (*(*__slate_slot_20)).prepare_fb }.unwrap()(*__slate_slot_31, *__slate_slot_30);
if *__slate_slot_29 != 0 {
break '__join_10;
}
} else {
let __ret_warn_on: i32 = ((*(*__slate_slot_20)).cleanup_fb != None) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("650: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 650b - ., 3; .popsection\n# {0}", const 650 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_prepare_planes_34, sym __slate_asm_string_drm_atomic_helper_prepare_planes_35, const 2848 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("651: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 651b - ., 4; .popsection\n# {0}", const 651 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if drm_core_check_feature(dev as *const drm_device, 1u32) {
*__slate_slot_29 = drm_gem_plane_helper_prepare_fb(*__slate_slot_31, *__slate_slot_30);
if *__slate_slot_29 != 0 {
break '__join_10;
}
}
}
}
let __v16: i32 = *__slate_slot_28;
*__slate_slot_28 = __v16.wrapping_add(1);
}
*__slate_slot_28 = 0;
'__join_21: {
while *__slate_slot_28 < (*(*state).dev).mode_config.num_total_plane {
if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_28 as isize)).ptr }.is_null() {
*__slate_slot_31 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_28 as isize)).ptr;
*__slate_slot_31;
*__slate_slot_30 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_28 as isize)).new_state;
*__slate_slot_30;
__v14 = true;
} else {
__v14 = false;
}
if __v14 {
funcs_2 = (*(*__slate_slot_31)).helper_private;
if (*funcs_2).begin_fb_access != None {
*__slate_slot_29 = unsafe { (*funcs_2).begin_fb_access }.unwrap()(*__slate_slot_31, *__slate_slot_30);
if *__slate_slot_29 != 0 {
break '__join_21;
}
}
}
let __v11: i32 = *__slate_slot_28;
*__slate_slot_28 = __v11.wrapping_add(1);
}
return 0;
}
*__slate_slot_27 = 0;
while *__slate_slot_27 < (*(*state).dev).mode_config.num_total_plane {
if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_27 as isize)).ptr }.is_null() {
*__slate_slot_31 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_27 as isize)).ptr;
*__slate_slot_31;
*__slate_slot_30 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_27 as isize)).new_state;
*__slate_slot_30;
__v9 = true;
} else {
__v9 = false;
}
if __v9 {
funcs_1 = (*(*__slate_slot_31)).helper_private;
if !(*__slate_slot_27 >= *__slate_slot_28) {
if (*funcs_1).end_fb_access != None {
unsafe { (*funcs_1).end_fb_access }.unwrap()(*__slate_slot_31, *__slate_slot_30);
}
}
}
let __v6: i32 = *__slate_slot_27;
*__slate_slot_27 = __v6.wrapping_add(1);
}
*__slate_slot_28 = *__slate_slot_27; // set i to upper limit to cleanup all planes
}
*__slate_slot_27 = 0;
while *__slate_slot_27 < (*(*state).dev).mode_config.num_total_plane {
if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_27 as isize)).ptr }.is_null() {
*__slate_slot_31 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_27 as isize)).ptr;
*__slate_slot_31;
*__slate_slot_30 = (*unsafe { (*state).planes }.wrapping_offset(*__slate_slot_27 as isize)).new_state;
*__slate_slot_30;
__v4 = true;
} else {
__v4 = false;
}
if __v4 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !(*__slate_slot_27 >= *__slate_slot_28) {
*__slate_slot_2 = (*(*__slate_slot_31)).helper_private;
if (*(*__slate_slot_2)).cleanup_fb != None {
unsafe { (*(*__slate_slot_2)).cleanup_fb }.unwrap()(*__slate_slot_31, *__slate_slot_30);
}
}
}
let __v1: i32 = *__slate_slot_27;
*__slate_slot_27 = __v1.wrapping_add(1);
}
return *__slate_slot_29;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_prepare_planes_652: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_prepare_planes)) };

/// drm_atomic_helper_unprepare_planes - release plane resources on aborts
/// @dev: DRM device
/// @state: atomic state object with old state structures
///
/// This function cleans up plane state, specifically framebuffers, from the
/// atomic state. It undoes the effects of drm_atomic_helper_prepare_planes()
/// when aborting an atomic commit. For cleaning up after a successful commit
/// use drm_atomic_helper_cleanup_planes().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_unprepare_planes")]
extern "C" fn drm_atomic_helper_unprepare_planes(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v0: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if (unsafe { (*funcs).end_fb_access }) != None {
unsafe { unsafe { (*funcs).end_fb_access }.unwrap()(plane, new_plane_state) };
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v5: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v5 {
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if (unsafe { (*funcs).cleanup_fb }) != None {
unsafe { unsafe { (*funcs).cleanup_fb }.unwrap()(plane, new_plane_state) };
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_unprepare_planes_653: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_unprepare_planes)) };

fn plane_crtc_active(state: *const drm_plane_state) -> bool {
!unsafe { (*state).crtc }.is_null() && unsafe { (*(*(*state).crtc).state).active }
}

/// drm_atomic_helper_commit_planes - commit plane state
/// @dev: DRM device
/// @state: atomic state object being committed
/// @flags: flags for committing plane state
///
/// This function commits the new plane state using the plane and atomic helper
/// functions for planes and CRTCs. It assumes that the atomic state has already
/// been pushed into the relevant object state pointers, since this step can no
/// longer fail.
///
/// It still requires the global state object @state to know which planes and
/// crtcs need to be updated though.
///
/// Note that this function does all plane updates across all CRTCs in one step.
/// If the hardware can't support this approach look at
/// drm_atomic_helper_commit_planes_on_crtc() instead.
///
/// Plane parameters can be updated by applications while the associated CRTC is
/// disabled. The DRM/KMS core will store the parameters in the plane state,
/// which will be available to the driver when the CRTC is turned on. As a result
/// most drivers don't need to be immediately notified of plane updates for a
/// disabled CRTC.
///
/// Unless otherwise needed, drivers are advised to set the ACTIVE_ONLY flag in
/// @flags in order not to receive plane update notifications related to a
/// disabled CRTC. This avoids the need to manually ignore plane updates in
/// driver code when the driver and/or hardware can't or just don't need to deal
/// with updates on disabled CRTCs, for example when supporting runtime PM.
///
/// Drivers may set the NO_DISABLE_AFTER_MODESET flag in @flags if the relevant
/// display controllers require to disable a CRTC's planes when the CRTC is
/// disabled. This function would skip the &drm_plane_helper_funcs.atomic_disable
/// call for a plane if the CRTC of the old plane state needs a modesetting
/// operation. Of course, the drivers need to disable the planes in their CRTC
/// disable callbacks since no one else would do that.
///
/// The drm_atomic_helper_commit() default implementation doesn't set the
/// ACTIVE_ONLY flag to most closely match the behaviour of the legacy helpers.
/// This should not be copied blindly by drivers.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_planes")]
extern "C" fn drm_atomic_helper_commit_planes(dev: *mut drm_device, state: *mut drm_atomic_commit, flags: u32) {
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut old_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let active_only: bool = (flags as u64) & BIT!(0i32) != 0;
let no_disable: bool = (flags as u64) & BIT!(1i32) != 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if !(funcs.is_null() || (unsafe { (*funcs).atomic_begin }) == None) {
if !(active_only && !unsafe { (*new_crtc_state).active }) {
unsafe { unsafe { (*funcs).atomic_begin }.unwrap()(crtc, state) };
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
'__slate_continue_9: {
let __v6: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v6 {
let mut disabling: bool = false;
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if !funcs.is_null() {
disabling = drm_atomic_plane_disabling(old_plane_state, new_plane_state);
if active_only {
// Skip planes related to inactive CRTCs. If the plane
// is enabled use the state of the current CRTC. If the
// plane is being disabled use the state of the old
// CRTC to avoid skipping planes being disabled on an
// active CRTC.
let __v10: bool = !disabling && !plane_crtc_active(new_plane_state as *const drm_plane_state);
if __v10 {
break '__slate_continue_9;
}
let __v11: bool = disabling && !plane_crtc_active(old_plane_state as *const drm_plane_state);
if __v11 {
break '__slate_continue_9;
}
}
// Special-case disabling the plane if drivers support it.
if disabling && (unsafe { (*funcs).atomic_disable }) != None {
let crtc_state: *mut drm_crtc_state = unsafe { (*(*old_plane_state).crtc).state };
if !(drm_atomic_crtc_needs_modeset(crtc_state as *const drm_crtc_state) && no_disable) {
unsafe { unsafe { (*funcs).atomic_disable }.unwrap()(plane, state) };
}
} else {
if !unsafe { (*new_plane_state).crtc }.is_null() || disabling {
unsafe { unsafe { (*funcs).atomic_update }.unwrap()(plane, state) };
if !disabling && (unsafe { (*funcs).atomic_enable }) != None {
if drm_atomic_plane_enabling(old_plane_state, new_plane_state) {
unsafe { unsafe { (*funcs).atomic_enable }.unwrap()(plane, state) };
}
}
}
}
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v14: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v14 {
let funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if !(funcs.is_null() || (unsafe { (*funcs).atomic_flush }) == None) {
if !(active_only && !unsafe { (*new_crtc_state).active }) {
unsafe { unsafe { (*funcs).atomic_flush }.unwrap()(crtc, state) };
}
}
}
i = i.wrapping_add(1);
}
// Signal end of framebuffer access here before hw_done. After hw_done,
// a later commit might have already released the plane state.
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v20: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v20 {
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if (unsafe { (*funcs).end_fb_access }) != None {
unsafe { unsafe { (*funcs).end_fb_access }.unwrap()(plane, old_plane_state) };
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_planes_654: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, u32)>, *mut ()>(Some(drm_atomic_helper_commit_planes)) };

/// drm_atomic_helper_commit_planes_on_crtc - commit plane state for a CRTC
/// @old_crtc_state: atomic state object with the old CRTC state
///
/// This function commits the new plane state using the plane and atomic helper
/// functions for planes on the specific CRTC. It assumes that the atomic state
/// has already been pushed into the relevant object state pointers, since this
/// step can no longer fail.
///
/// This function is useful when plane updates should be done CRTC-by-CRTC
/// instead of one global step like drm_atomic_helper_commit_planes() does.
///
/// This function can only be savely used when planes are not allowed to move
/// between different CRTCs because this function doesn't handle inter-CRTC
/// dependencies. Callers need to ensure that either no such dependencies exist,
/// resolve them through ordering of commit calls or through some other means.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_planes_on_crtc")]
extern "C" fn drm_atomic_helper_commit_planes_on_crtc(old_crtc_state: *mut drm_crtc_state) {
let crtc: *mut drm_crtc = unsafe { (*old_crtc_state).crtc };
let old_state: *mut drm_atomic_commit = unsafe { (*old_crtc_state).state };
let new_crtc_state: *mut drm_crtc_state = drm_atomic_get_new_crtc_state(old_state as *const drm_atomic_commit, crtc);
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut plane_mask: u32 = unsafe { (*old_crtc_state).plane_mask };
plane_mask |= unsafe { (*new_crtc_state).plane_mask };
let crtc_funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
if !crtc_funcs.is_null() && (unsafe { (*crtc_funcs).atomic_begin }) != None {
unsafe { unsafe { (*crtc_funcs).atomic_begin }.unwrap()(crtc, old_state) };
}
plane = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*(*crtc).dev).mode_config.plane_list) });
while list_entry_is_head!(plane, unsafe { core::ptr::addr_of_mut!((*(*crtc).dev).mode_config.plane_list) }, head) == 0 {
if plane_mask & drm_plane_mask(plane as *const drm_plane) != 0 {
let old_plane_state: *mut drm_plane_state = drm_atomic_get_old_plane_state(old_state as *const drm_atomic_commit, plane);
let new_plane_state: *mut drm_plane_state = drm_atomic_get_new_plane_state(old_state as *const drm_atomic_commit, plane);
let mut disabling: bool = false;
let plane_funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if !(old_plane_state.is_null() || plane_funcs.is_null()) {
let mut __v4: i64;
let __ret_warn_on: i32 = (!unsafe { (*new_plane_state).crtc }.is_null() && (unsafe { (*new_plane_state).crtc }) != crtc) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("655: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 655b - ., 3; .popsection\n# {0}", const 655 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_commit_planes_on_crtc_5, sym __slate_asm_string_drm_atomic_helper_commit_planes_on_crtc_6, const 3129 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("656: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 656b - ., 4; .popsection\n# {0}", const 656 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
disabling = drm_atomic_plane_disabling(old_plane_state, new_plane_state);
if disabling && (unsafe { (*plane_funcs).atomic_disable }) != None {
unsafe { unsafe { (*plane_funcs).atomic_disable }.unwrap()(plane, old_state) };
} else {
if !unsafe { (*new_plane_state).crtc }.is_null() || disabling {
unsafe { unsafe { (*plane_funcs).atomic_update }.unwrap()(plane, old_state) };
if !disabling && (unsafe { (*plane_funcs).atomic_enable }) != None {
if drm_atomic_plane_enabling(old_plane_state, new_plane_state) {
unsafe { unsafe { (*plane_funcs).atomic_enable }.unwrap()(plane, old_state) };
}
}
}
}
}
}
plane = list_next_entry__4!(plane, head);
}
if !crtc_funcs.is_null() && (unsafe { (*crtc_funcs).atomic_flush }) != None {
unsafe { unsafe { (*crtc_funcs).atomic_flush }.unwrap()(crtc, old_state) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_planes_on_crtc_657: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_crtc_state)>, *mut ()>(Some(drm_atomic_helper_commit_planes_on_crtc)) };

/// drm_atomic_helper_disable_planes_on_crtc - helper to disable CRTC's planes
/// @old_crtc_state: atomic state object with the old CRTC state
/// @atomic: if set, synchronize with CRTC's atomic_begin/flush hooks
///
/// Disables all planes associated with the given CRTC. This can be
/// used for instance in the CRTC helper atomic_disable callback to disable
/// all planes.
///
/// If the atomic-parameter is set the function calls the CRTC's
/// atomic_begin hook before and atomic_flush hook after disabling the
/// planes.
///
/// It is a bug to call this function without having implemented the
/// &drm_plane_helper_funcs.atomic_disable plane hook.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_disable_planes_on_crtc")]
extern "C" fn drm_atomic_helper_disable_planes_on_crtc(old_crtc_state: *mut drm_crtc_state, atomic: bool) {
let crtc: *mut drm_crtc = unsafe { (*old_crtc_state).crtc };
let crtc_funcs: *const drm_crtc_helper_funcs = unsafe { (*crtc).helper_private };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
if atomic && !crtc_funcs.is_null() && (unsafe { (*crtc_funcs).atomic_begin }) != None {
unsafe { unsafe { (*crtc_funcs).atomic_begin }.unwrap()(crtc, core::ptr::null_mut::<drm_atomic_commit>()) };
}
plane = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*(*(*old_crtc_state).state).dev).mode_config.plane_list) });
while list_entry_is_head!(plane, unsafe { core::ptr::addr_of_mut!((*(*(*old_crtc_state).state).dev).mode_config.plane_list) }, head) == 0 {
if (unsafe { (*old_crtc_state).plane_mask }) & drm_plane_mask(plane as *const drm_plane) != 0 {
let plane_funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if !plane_funcs.is_null() {
let mut __v2: i64;
let __ret_warn_on: i32 = ((unsafe { (*plane_funcs).atomic_disable }) == None) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("658: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 658b - ., 3; .popsection\n# {0}", const 658 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_disable_planes_on_crtc_3, sym __slate_asm_string_drm_atomic_helper_disable_planes_on_crtc_4, const 3185 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("659: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 659b - ., 4; .popsection\n# {0}", const 659 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if (unsafe { (*plane_funcs).atomic_disable }) != None {
unsafe { unsafe { (*plane_funcs).atomic_disable }.unwrap()(plane, core::ptr::null_mut::<drm_atomic_commit>()) };
}
}
}
plane = list_next_entry__4!(plane, head);
}
if atomic && !crtc_funcs.is_null() && (unsafe { (*crtc_funcs).atomic_flush }) != None {
unsafe { unsafe { (*crtc_funcs).atomic_flush }.unwrap()(crtc, core::ptr::null_mut::<drm_atomic_commit>()) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_disable_planes_on_crtc_660: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_crtc_state, bool)>, *mut ()>(Some(drm_atomic_helper_disable_planes_on_crtc)) };

/// drm_atomic_helper_cleanup_planes - cleanup plane resources after commit
/// @dev: DRM device
/// @state: atomic state object being committed
///
/// This function cleans up plane state, specifically framebuffers, from the old
/// configuration. Hence the old configuration must be perserved in @state to
/// be able to call this function.
///
/// This function may not be called on the new state when the atomic update
/// fails at any point after calling drm_atomic_helper_prepare_planes(). Use
/// drm_atomic_helper_unprepare_planes() in this case.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_cleanup_planes")]
extern "C" fn drm_atomic_helper_cleanup_planes(dev: *mut drm_device, state: *mut drm_atomic_commit) {
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut old_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v0: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v0 {
let funcs: *const drm_plane_helper_funcs = unsafe { (*plane).helper_private };
if (unsafe { (*funcs).cleanup_fb }) != None {
unsafe { unsafe { (*funcs).cleanup_fb }.unwrap()(plane, old_plane_state) };
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_cleanup_planes_661: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit)>, *mut ()>(Some(drm_atomic_helper_cleanup_planes)) };

/// drm_atomic_helper_swap_state - store atomic state into current sw state
/// @state: atomic state
/// @stall: stall for preceding commits
///
/// This function stores the atomic state into the current state pointers in all
/// driver objects. It should be called after all failing steps have been done
/// and succeeded, but before the actual hardware state is committed.
///
/// For cleanup and error recovery the current state for all changed objects will
/// be swapped into @state.
///
/// With that sequence it fits perfectly into the plane prepare/cleanup sequence:
///
/// 1. Call drm_atomic_helper_prepare_planes() with the staged atomic state.
///
/// 2. Do any other steps that might fail.
///
/// 3. Put the staged state into the current state pointers with this function.
///
/// 4. Actually commit the hardware state.
///
/// 5. Call drm_atomic_helper_cleanup_planes() with @state, which since step 3
/// contains the old state. Also do any other cleanup required with that state.
///
/// @stall must be set when nonblocking commits for this driver directly access
/// the &drm_plane.state, &drm_crtc.state or &drm_connector.state pointer. With
/// the current atomic helpers this is almost always the case, since the helpers
/// don't pass the right state structures to the callbacks.
///
/// Returns:
/// Returns 0 on success. Can return -ERESTARTSYS when @stall is true and the
/// waiting for the previous commits has been interrupted.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_swap_state")]
extern "C" fn drm_atomic_helper_swap_state(state: *mut drm_atomic_commit, stall: bool) -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
let flags: u64 = 0;
let mut connector: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut old_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut new_conn_state: *mut drm_connector_state = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
let mut old_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut new_crtc_state: *mut drm_crtc_state = unsafe { core::mem::zeroed() };
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut old_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut new_plane_state: *mut drm_plane_state = unsafe { core::mem::zeroed() };
let mut colorop: *mut drm_colorop = unsafe { core::mem::zeroed() };
let mut old_colorop_state: *mut drm_colorop_state = unsafe { core::mem::zeroed() };
let mut new_colorop_state: *mut drm_colorop_state = unsafe { core::mem::zeroed() };
let mut commit: *mut drm_crtc_commit = unsafe { core::mem::zeroed() };
let mut obj: *mut drm_private_obj = unsafe { core::mem::zeroed() };
let mut old_obj_state: *mut drm_private_state = unsafe { core::mem::zeroed() };
let mut new_obj_state: *mut drm_private_state = unsafe { core::mem::zeroed() };
if stall {
// We have to stall for hw_done here before
// drm_atomic_helper_wait_for_dependencies() because flip
// depth > 1 is not yet supported by all drivers. As long as
// obj->state is directly dereferenced anywhere in the drivers
// atomic_commit_tail function, then it's unsafe to swap state
// before drm_atomic_helper_commit_hw_done() is called.
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v0: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v0 {
commit = unsafe { (*old_crtc_state).commit };
if !commit.is_null() {
ret = unsafe { wait_for_completion_interruptible(core::ptr::addr_of_mut!((*commit).hw_done)) };
if ret != 0 {
return ret;
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v5: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v5 {
commit = unsafe { (*old_conn_state).commit };
if !commit.is_null() {
ret = unsafe { wait_for_completion_interruptible(core::ptr::addr_of_mut!((*commit).hw_done)) };
if ret != 0 {
return ret;
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v10: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
true
} else {
false
};
if __v10 {
commit = unsafe { (*old_plane_state).commit };
if !commit.is_null() {
ret = unsafe { wait_for_completion_interruptible(core::ptr::addr_of_mut!((*commit).hw_done)) };
if ret != 0 {
return ret;
}
}
}
i = i.wrapping_add(1);
}
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v15: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
old_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state };
new_conn_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v15 {
let mut __v18: i64;
let __ret_warn_on: i32 = ((unsafe { (*connector).state }) != old_conn_state) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("662: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 662b - ., 3; .popsection\n# {0}", const 662 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_swap_state_19, sym __slate_asm_string_drm_atomic_helper_swap_state_20, const 3320 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("663: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 663b - ., 4; .popsection\n# {0}", const 663 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v18 = (__ret_warn_on != 0) as i64;
unsafe {
(*old_conn_state).state = state;
}
unsafe {
(*new_conn_state).state = core::ptr::null_mut::<drm_atomic_commit>();
}
unsafe {
(*unsafe { (*state).connectors }.wrapping_offset(i as isize)).state_to_destroy = old_conn_state;
}
unsafe {
(*connector).state = new_conn_state;
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v26: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
old_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state };
new_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v26 {
let mut __v29: i64;
let __ret_warn_on: i32 = ((unsafe { (*crtc).state }) != old_crtc_state) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("664: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 664b - ., 3; .popsection\n# {0}", const 664 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_swap_state_30, sym __slate_asm_string_drm_atomic_helper_swap_state_31, const 3330 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("665: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 665b - ., 4; .popsection\n# {0}", const 665 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v29 = (__ret_warn_on != 0) as i64;
unsafe {
(*old_crtc_state).state = state;
}
unsafe {
(*new_crtc_state).state = core::ptr::null_mut::<drm_atomic_commit>();
}
unsafe {
(*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).state_to_destroy = old_crtc_state;
}
unsafe {
(*crtc).state = new_crtc_state;
}
if !unsafe { (*new_crtc_state).commit }.is_null() {
spin_lock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
list_add(unsafe { core::ptr::addr_of_mut!((*(*new_crtc_state).commit).commit_entry) }, unsafe { core::ptr::addr_of_mut!((*crtc).commit_list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*crtc).commit_lock) });
unsafe {
(*(*new_crtc_state).commit).event = core::ptr::null_mut::<drm_pending_vblank_event>();
}
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_colorop } {
let __v37: bool = if !unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).ptr }.is_null() {
colorop = unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).ptr };
old_colorop_state = unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).old_state };
new_colorop_state = unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v37 {
let mut __v40: i64;
let __ret_warn_on: i32 = ((unsafe { (*colorop).state }) != old_colorop_state) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("666: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 666b - ., 3; .popsection\n# {0}", const 666 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_swap_state_41, sym __slate_asm_string_drm_atomic_helper_swap_state_42, const 3349 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("667: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 667b - ., 4; .popsection\n# {0}", const 667 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v40 = (__ret_warn_on != 0) as i64;
unsafe {
(*old_colorop_state).state = state;
}
unsafe {
(*new_colorop_state).state = core::ptr::null_mut::<drm_atomic_commit>();
}
unsafe {
(*unsafe { (*state).colorops }.wrapping_offset(i as isize)).state = old_colorop_state;
}
unsafe {
(*colorop).state = new_colorop_state;
}
}
i = i.wrapping_add(1);
}
drm_panic_lock(unsafe { (*state).dev }, flags);
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v48: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
old_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state };
new_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v48 {
let mut __v51: i64;
let __ret_warn_on: i32 = ((unsafe { (*plane).state }) != old_plane_state) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("668: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 668b - ., 3; .popsection\n# {0}", const 668 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_swap_state_52, sym __slate_asm_string_drm_atomic_helper_swap_state_53, const 3360 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("669: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 669b - ., 4; .popsection\n# {0}", const 669 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v51 = (__ret_warn_on != 0) as i64;
unsafe {
(*old_plane_state).state = state;
}
unsafe {
(*new_plane_state).state = core::ptr::null_mut::<drm_atomic_commit>();
}
unsafe {
(*unsafe { (*state).planes }.wrapping_offset(i as isize)).state_to_destroy = old_plane_state;
}
unsafe {
(*plane).state = new_plane_state;
}
}
i = i.wrapping_add(1);
}
drm_panic_unlock(unsafe { (*state).dev }, flags);
i = 0;
loop {
let __v64: bool = if i < unsafe { (*state).num_private_objs } {
obj = unsafe { (*unsafe { (*state).private_objs }.wrapping_offset(i as isize)).ptr };
old_obj_state = unsafe { (*unsafe { (*state).private_objs }.wrapping_offset(i as isize)).old_state };
new_obj_state = unsafe { (*unsafe { (*state).private_objs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if !__v64 {
break;
}
let mut __v59: i64;
let __ret_warn_on: i32 = ((unsafe { (*obj).state }) != old_obj_state) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("670: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 670b - ., 3; .popsection\n# {0}", const 670 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_swap_state_60, sym __slate_asm_string_drm_atomic_helper_swap_state_61, const 3371 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("671: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 671b - ., 4; .popsection\n# {0}", const 671 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v59 = (__ret_warn_on != 0) as i64;
unsafe {
(*old_obj_state).state = state;
}
unsafe {
(*new_obj_state).state = core::ptr::null_mut::<drm_atomic_commit>();
}
unsafe {
(*unsafe { (*state).private_objs }.wrapping_offset(i as isize)).state_to_destroy = old_obj_state;
}
unsafe {
(*obj).state = new_obj_state;
}
i = i.wrapping_add(1);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_swap_state_672: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, bool) -> i32>, *mut ()>(Some(drm_atomic_helper_swap_state)) };

/// drm_atomic_helper_update_plane - Helper for primary plane update using atomic
/// @plane: plane object to update
/// @crtc: owning CRTC of owning plane
/// @fb: framebuffer to flip onto plane
/// @crtc_x: x offset of primary plane on @crtc
/// @crtc_y: y offset of primary plane on @crtc
/// @crtc_w: width of primary plane rectangle on @crtc
/// @crtc_h: height of primary plane rectangle on @crtc
/// @src_x: x offset of @fb for panning
/// @src_y: y offset of @fb for panning
/// @src_w: width of source rectangle in @fb
/// @src_h: height of source rectangle in @fb
/// @ctx: lock acquire context
///
/// Provides a default plane update handler using the atomic driver interface.
///
/// RETURNS:
/// Zero on success, error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_update_plane")]
extern "C" fn drm_atomic_helper_update_plane(plane: *mut drm_plane, crtc: *mut drm_crtc, fb: *mut drm_framebuffer, crtc_x: i32, crtc_y: i32, crtc_w: u32, crtc_h: u32, src_x: u32, src_y: u32, src_w: u32, src_h: u32, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut plane_state: core::mem::MaybeUninit<*mut drm_plane_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut drm_plane_state = core::ptr::addr_of_mut!(plane_state) as *mut *mut drm_plane_state;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_2 = drm_atomic_commit_alloc((*plane).dev);
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_2)).acquire_ctx = ctx;
*__slate_slot_1 = drm_atomic_get_plane_state(*__slate_slot_2, plane);
if IS_ERR(*__slate_slot_1 as *const ()) {
ret = PTR_ERR(*__slate_slot_1 as *const ()) as i32;
} else {
ret = drm_atomic_set_crtc_for_plane(*__slate_slot_1, crtc);
if ret == 0 {
drm_atomic_set_fb_for_plane(*__slate_slot_1, fb);
(*(*__slate_slot_1)).crtc_x = crtc_x;
(*(*__slate_slot_1)).crtc_y = crtc_y;
(*(*__slate_slot_1)).crtc_w = crtc_w;
(*(*__slate_slot_1)).crtc_h = crtc_h;
(*(*__slate_slot_1)).src_x = src_x;
(*(*__slate_slot_1)).src_y = src_y;
(*(*__slate_slot_1)).src_w = src_w;
(*(*__slate_slot_1)).src_h = src_h;
if plane == (*crtc).cursor {
let __t0 = true;
(*(*__slate_slot_2)).__slate_bits_0.__set_legacy_cursor_update(__t0);
}
ret = drm_atomic_commit(*__slate_slot_2);
}
}
drm_atomic_commit_put(*__slate_slot_2);
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_update_plane_673: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_crtc, *mut drm_framebuffer, i32, i32, u32, u32, u32, u32, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_update_plane)) };

/// drm_atomic_helper_disable_plane - Helper for primary plane disable using atomic
/// @plane: plane to disable
/// @ctx: lock acquire context
///
/// Provides a default plane disable handler using the atomic driver interface.
///
/// RETURNS:
/// Zero on success, error code on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_disable_plane")]
extern "C" fn drm_atomic_helper_disable_plane(plane: *mut drm_plane, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut plane_state: core::mem::MaybeUninit<*mut drm_plane_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut drm_plane_state = core::ptr::addr_of_mut!(plane_state) as *mut *mut drm_plane_state;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_2 = drm_atomic_commit_alloc((*plane).dev);
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_2)).acquire_ctx = ctx;
*__slate_slot_1 = drm_atomic_get_plane_state(*__slate_slot_2, plane);
if IS_ERR(*__slate_slot_1 as *const ()) {
ret = PTR_ERR(*__slate_slot_1 as *const ()) as i32;
} else {
if !unsafe { (*(*__slate_slot_1)).crtc }.is_null() && (*(*(*__slate_slot_1)).crtc).cursor == plane {
let __t0 = true;
(*(*(*__slate_slot_1)).state).__slate_bits_0.__set_legacy_cursor_update(__t0);
}
ret = __drm_atomic_helper_disable_plane(plane, *__slate_slot_1);
if ret == 0 {
ret = drm_atomic_commit(*__slate_slot_2);
}
}
drm_atomic_commit_put(*__slate_slot_2);
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_disable_plane_674: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_disable_plane)) };

/// drm_atomic_helper_set_config - set a new config from userspace
/// @set: mode set configuration
/// @ctx: lock acquisition context
///
/// Provides a default CRTC set_config handler using the atomic driver interface.
///
/// NOTE: For backwards compatibility with old userspace this automatically
/// resets the "link-status" property to GOOD, to force any link
/// re-training. The SETCRTC ioctl does not define whether an update does
/// need a full modeset or just a plane update, hence we're allowed to do
/// that. See also drm_connector_set_link_status_property().
///
/// Returns:
/// Returns 0 on success, negative errno numbers on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_set_config")]
extern "C" fn drm_atomic_helper_set_config(set: *mut drm_mode_set, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let crtc: *mut drm_crtc = (*set).crtc;
ret = 0;
*__slate_slot_2 = drm_atomic_commit_alloc((*crtc).dev);
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_2)).acquire_ctx = ctx;
ret = __drm_atomic_helper_set_config(set, *__slate_slot_2);
if ret == 0 {
ret = handle_conflicting_encoders(*__slate_slot_2, true);
if ret == 0 {
ret = drm_atomic_commit(*__slate_slot_2);
}
}
drm_atomic_commit_put(*__slate_slot_2);
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_set_config_675: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_mode_set, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_set_config)) };

/// drm_atomic_helper_disable_all - disable all currently active outputs
/// @dev: DRM device
/// @ctx: lock acquisition context
///
/// Loops through all connectors, finding those that aren't turned off and then
/// turns them off by setting their DPMS mode to OFF and deactivating the CRTC
/// that they are connected to.
///
/// This is used for example in suspend/resume to disable all currently active
/// functions when suspending. If you just want to shut down everything at e.g.
/// driver unload, look at drm_atomic_helper_shutdown().
///
/// Note that if callers haven't already acquired all modeset locks this might
/// return -EDEADLK, which must be handled by calling drm_modeset_backoff().
///
/// Returns:
/// 0 on success or a negative error code on failure.
///
/// See also:
/// drm_atomic_helper_suspend(), drm_atomic_helper_resume() and
/// drm_atomic_helper_shutdown().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_disable_all")]
extern "C" fn drm_atomic_helper_disable_all(dev: *mut drm_device, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut crtc: core::mem::MaybeUninit<*mut drm_crtc> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut drm_crtc = core::ptr::addr_of_mut!(crtc) as *mut *mut drm_crtc;
let mut crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(crtc_state) as *mut *mut drm_crtc_state;
let mut plane: core::mem::MaybeUninit<*mut drm_plane> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut drm_plane = core::ptr::addr_of_mut!(plane) as *mut *mut drm_plane;
let mut plane_state: core::mem::MaybeUninit<*mut drm_plane_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut drm_plane_state = core::ptr::addr_of_mut!(plane_state) as *mut *mut drm_plane_state;
let mut conn: core::mem::MaybeUninit<*mut drm_connector> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut drm_connector = core::ptr::addr_of_mut!(conn) as *mut *mut drm_connector;
let mut conn_state: core::mem::MaybeUninit<*mut drm_connector_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut drm_connector_state = core::ptr::addr_of_mut!(conn_state) as *mut *mut drm_connector_state;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_16 = drm_atomic_commit_alloc(dev);
if (*__slate_slot_16).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_16)).acquire_ctx = ctx;
*__slate_slot_10 = list_first_entry!(core::ptr::addr_of_mut!((*dev).mode_config.crtc_list));
'__join_0: {
'__join_24: {
while list_entry_is_head!(*__slate_slot_10, core::ptr::addr_of_mut!((*dev).mode_config.crtc_list), head) == 0 {
*__slate_slot_11 = drm_atomic_get_crtc_state(*__slate_slot_16, *__slate_slot_10);
if IS_ERR(*__slate_slot_11 as *const ()) {
break '__join_24;
} else {
(*(*__slate_slot_11)).active = false;
*__slate_slot_9 = drm_atomic_set_mode_prop_for_crtc(*__slate_slot_11, core::ptr::null_mut::<drm_property_blob>());
if *__slate_slot_9 < 0 {
break '__join_0;
} else {
*__slate_slot_9 = drm_atomic_add_affected_planes(*__slate_slot_16, *__slate_slot_10);
if *__slate_slot_9 < 0 {
break '__join_0;
} else {
*__slate_slot_9 = drm_atomic_add_affected_connectors(*__slate_slot_16, *__slate_slot_10);
if *__slate_slot_9 < 0 {
break '__join_0;
} else {
*__slate_slot_10 = list_next_entry__3!(*__slate_slot_10, head);
}
}
}
}
}
*__slate_slot_8 = 0;
while *__slate_slot_8 < (*(*__slate_slot_16)).num_connector {
if !unsafe { (*unsafe { (*(*__slate_slot_16)).connectors }.wrapping_offset(*__slate_slot_8 as isize)).ptr }.is_null() {
*__slate_slot_14 = (*unsafe { (*(*__slate_slot_16)).connectors }.wrapping_offset(*__slate_slot_8 as isize)).ptr;
*__slate_slot_14;
*__slate_slot_15 = (*unsafe { (*(*__slate_slot_16)).connectors }.wrapping_offset(*__slate_slot_8 as isize)).new_state;
*__slate_slot_15;
__v7 = true;
} else {
__v7 = false;
}
if __v7 {
*__slate_slot_9 = drm_atomic_set_crtc_for_connector(*__slate_slot_15, core::ptr::null_mut::<drm_crtc>());
if *__slate_slot_9 < 0 {
break '__join_0;
}
}
let __v5: i32 = *__slate_slot_8;
*__slate_slot_8 = __v5.wrapping_add(1);
}
*__slate_slot_8 = 0;
while *__slate_slot_8 < (*(*(*__slate_slot_16)).dev).mode_config.num_total_plane {
if !unsafe { (*unsafe { (*(*__slate_slot_16)).planes }.wrapping_offset(*__slate_slot_8 as isize)).ptr }.is_null() {
*__slate_slot_12 = (*unsafe { (*(*__slate_slot_16)).planes }.wrapping_offset(*__slate_slot_8 as isize)).ptr;
*__slate_slot_12;
*__slate_slot_13 = (*unsafe { (*(*__slate_slot_16)).planes }.wrapping_offset(*__slate_slot_8 as isize)).new_state;
*__slate_slot_13;
__v3 = true;
} else {
__v3 = false;
}
if __v3 {
*__slate_slot_9 = drm_atomic_set_crtc_for_plane(*__slate_slot_13, core::ptr::null_mut::<drm_crtc>());
if *__slate_slot_9 < 0 {
break '__join_0;
} else {
drm_atomic_set_fb_for_plane(*__slate_slot_13, core::ptr::null_mut::<drm_framebuffer>());
}
}
let __v1: i32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
*__slate_slot_9 = drm_atomic_commit(*__slate_slot_16);
break '__join_0;
}
*__slate_slot_9 = PTR_ERR(*__slate_slot_11 as *const ()) as i32;
}
drm_atomic_commit_put(*__slate_slot_16);
return *__slate_slot_9;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_disable_all_676: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_disable_all)) };

/// drm_atomic_helper_reset_crtc - reset the active outputs of a CRTC
/// @crtc: DRM CRTC
/// @ctx: lock acquisition context
///
/// Reset the active outputs by indicating that connectors have changed.
/// This implies a reset of all active components available between the CRTC and
/// connectors.
///
/// A variant of this function exists with
/// drm_bridge_helper_reset_crtc(), dedicated to bridges.
///
/// NOTE: This relies on resetting &drm_crtc_state.connectors_changed.
/// For drivers which optimize out unnecessary modesets this will result in
/// a no-op commit, achieving nothing.
///
/// Returns:
/// 0 on success or a negative error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_reset_crtc")]
extern "C" fn drm_atomic_helper_reset_crtc(crtc: *mut drm_crtc, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(crtc_state) as *mut *mut drm_crtc_state;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_2 = drm_atomic_commit_alloc((*crtc).dev);
if (*__slate_slot_2).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_2)).acquire_ctx = ctx;
*__slate_slot_1 = drm_atomic_get_crtc_state(*__slate_slot_2, crtc);
if IS_ERR(*__slate_slot_1 as *const ()) {
*__slate_slot_0 = PTR_ERR(*__slate_slot_1 as *const ()) as i32;
} else {
let __t0 = true;
(*(*__slate_slot_1)).__slate_bits_0.__set_connectors_changed(__t0);
*__slate_slot_0 = drm_atomic_commit(*__slate_slot_2);
}
drm_atomic_commit_put(*__slate_slot_2);
return *__slate_slot_0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_reset_crtc_677: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_reset_crtc)) };

/// drm_atomic_helper_shutdown - shutdown all CRTC
/// @dev: DRM device
///
/// This shuts down all CRTC, which is useful for driver unloading. Shutdown on
/// suspend should instead be handled with drm_atomic_helper_suspend(), since
/// that also takes a snapshot of the modeset state to be restored on resume.
///
/// This is just a convenience wrapper around drm_atomic_helper_disable_all(),
/// and it is the atomic version of drm_helper_force_disable_all().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_shutdown")]
extern "C" fn drm_atomic_helper_shutdown(dev: *mut drm_device) {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut ctx: core::mem::MaybeUninit<drm_modeset_acquire_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut drm_modeset_acquire_ctx = core::ptr::addr_of_mut!(ctx) as *mut drm_modeset_acquire_ctx;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if dev.is_null() {
return;
} else {
if !drm_drv_uses_atomic_modeset(dev) {
mutex_lock(core::ptr::addr_of_mut!((*dev).mode_config.mutex));
}
drm_modeset_acquire_init(core::ptr::addr_of_mut!(*__slate_slot_1), 0u32);
loop {
*__slate_slot_0 = drm_modeset_lock_all_ctx(dev, core::ptr::addr_of_mut!(*__slate_slot_1));
if *__slate_slot_0 == 0 {
*__slate_slot_0 = drm_atomic_helper_disable_all(dev, core::ptr::addr_of_mut!(*__slate_slot_1));
if *__slate_slot_0 != 0 {
while false {
}
_dev_err(__drm_to_dev(dev as *const drm_device) as *const device, c"[drm] *ERROR* Disabling all crtc's during unload failed with %i\n".as_ptr() as *const u8, *__slate_slot_0);
}
}
if *__slate_slot_0 == 35i32.wrapping_neg() {
*__slate_slot_0 = drm_modeset_backoff(core::ptr::addr_of_mut!(*__slate_slot_1));
if *__slate_slot_0 != 0 {
break;
}
} else {
break;
}
}
drm_modeset_drop_locks(core::ptr::addr_of_mut!(*__slate_slot_1));
drm_modeset_acquire_fini(core::ptr::addr_of_mut!(*__slate_slot_1));
if !drm_drv_uses_atomic_modeset(dev) {
mutex_unlock(core::ptr::addr_of_mut!((*dev).mode_config.mutex));
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_shutdown_678: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device)>, *mut ()>(Some(drm_atomic_helper_shutdown)) };

/// drm_atomic_helper_duplicate_state - duplicate an atomic state object
/// @dev: DRM device
/// @ctx: lock acquisition context
///
/// Makes a copy of the current atomic state by looping over all objects and
/// duplicating their respective states. This is used for example by suspend/
/// resume support code to save the state prior to suspend such that it can
/// be restored upon resume.
///
/// Note that this treats atomic state as persistent between save and restore.
/// Drivers must make sure that this is possible and won't result in confusion
/// or erroneous behaviour.
///
/// Note that if callers haven't already acquired all modeset locks this might
/// return -EDEADLK, which must be handled by calling drm_modeset_backoff().
///
/// Returns:
/// A pointer to the copy of the atomic state object on success or an
/// ERR_PTR()-encoded error code on failure.
///
/// See also:
/// drm_atomic_helper_suspend(), drm_atomic_helper_resume()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_duplicate_state")]
extern "C" fn drm_atomic_helper_duplicate_state(dev: *mut drm_device, ctx: *mut drm_modeset_acquire_ctx) -> *mut drm_atomic_commit {
let mut conn_state: core::mem::MaybeUninit<*mut drm_connector_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut drm_connector_state = core::ptr::addr_of_mut!(conn_state) as *mut *mut drm_connector_state;
let mut plane_state: core::mem::MaybeUninit<*mut drm_plane_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut drm_plane_state = core::ptr::addr_of_mut!(plane_state) as *mut *mut drm_plane_state;
let mut crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(crtc_state) as *mut *mut drm_crtc_state;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut crtc: core::mem::MaybeUninit<*mut drm_crtc> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut drm_crtc = core::ptr::addr_of_mut!(crtc) as *mut *mut drm_crtc;
let mut plane: core::mem::MaybeUninit<*mut drm_plane> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut drm_plane = core::ptr::addr_of_mut!(plane) as *mut *mut drm_plane;
let mut conn_iter: core::mem::MaybeUninit<drm_connector_list_iter> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut drm_connector_list_iter = core::ptr::addr_of_mut!(conn_iter) as *mut drm_connector_list_iter;
let mut conn: core::mem::MaybeUninit<*mut drm_connector> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut drm_connector = core::ptr::addr_of_mut!(conn) as *mut *mut drm_connector;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
err = 0;
*__slate_slot_9 = drm_atomic_commit_alloc(dev);
if (*__slate_slot_9).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut drm_atomic_commit;
} else {
(*(*__slate_slot_9)).acquire_ctx = ctx;
let __t0 = true;
(*(*__slate_slot_9)).__slate_bits_0.__set_duplicated(__t0);
*__slate_slot_5 = list_first_entry!(core::ptr::addr_of_mut!((*dev).mode_config.crtc_list));
'__join_2: {
'__join_17: {
while list_entry_is_head!(*__slate_slot_5, core::ptr::addr_of_mut!((*dev).mode_config.crtc_list), head) == 0 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = drm_atomic_get_crtc_state(*__slate_slot_9, *__slate_slot_5);
if IS_ERR(*__slate_slot_3 as *const ()) {
break '__join_17;
} else {
*__slate_slot_5 = list_next_entry__3!(*__slate_slot_5, head);
}
}
*__slate_slot_6 = list_first_entry__3!(core::ptr::addr_of_mut!((*dev).mode_config.plane_list));
'__join_12: {
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*dev).mode_config.plane_list), head) == 0 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = drm_atomic_get_plane_state(*__slate_slot_9, *__slate_slot_6);
if IS_ERR(*__slate_slot_2 as *const ()) {
break '__join_12;
} else {
if !unsafe { (*(*__slate_slot_2)).color_pipeline }.is_null() {
err = drm_atomic_add_affected_colorops(*__slate_slot_9, *__slate_slot_6);
if err != 0 {
break '__join_2;
}
}
*__slate_slot_6 = list_next_entry__4!(*__slate_slot_6, head);
}
}
drm_connector_list_iter_begin(dev, core::ptr::addr_of_mut!(*__slate_slot_7));
'__join_5: {
loop {
let __v0: *mut drm_connector = drm_connector_list_iter_next(core::ptr::addr_of_mut!(*__slate_slot_7));
*__slate_slot_8 = __v0;
if !__v0.is_null() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = drm_atomic_get_connector_state(*__slate_slot_9, *__slate_slot_8);
if IS_ERR(*__slate_slot_1 as *const ()) {
break '__join_5;
}
} else {
break;
}
}
drm_connector_list_iter_end(core::ptr::addr_of_mut!(*__slate_slot_7));
// clear the acquire context so that it isn't accidentally reused
(*(*__slate_slot_9)).acquire_ctx = core::ptr::null_mut::<drm_modeset_acquire_ctx>();
break '__join_2;
}
err = PTR_ERR(*__slate_slot_1 as *const ()) as i32;
drm_connector_list_iter_end(core::ptr::addr_of_mut!(*__slate_slot_7));
break '__join_2;
}
err = PTR_ERR(*__slate_slot_2 as *const ()) as i32;
break '__join_2;
}
err = PTR_ERR(*__slate_slot_3 as *const ()) as i32;
}
if err < 0 {
drm_atomic_commit_put(*__slate_slot_9);
*__slate_slot_9 = ERR_PTR(err as i64) as *mut drm_atomic_commit;
}
return *__slate_slot_9;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_duplicate_state_679: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_modeset_acquire_ctx) -> *mut drm_atomic_commit>, *mut ()>(Some(drm_atomic_helper_duplicate_state)) };

/// drm_atomic_helper_suspend - subsystem-level suspend helper
/// @dev: DRM device
///
/// Duplicates the current atomic state, disables all active outputs and then
/// returns a pointer to the original atomic state to the caller. Drivers can
/// pass this pointer to the drm_atomic_helper_resume() helper upon resume to
/// restore the output configuration that was active at the time the system
/// entered suspend.
///
/// Note that it is potentially unsafe to use this. The atomic state object
/// returned by this function is assumed to be persistent. Drivers must ensure
/// that this holds true. Before calling this function, drivers must make sure
/// to suspend fbdev emulation so that nothing can be using the device.
///
/// Returns:
/// A pointer to a copy of the state before suspend on success or an ERR_PTR()-
/// encoded error code on failure. Drivers should store the returned atomic
/// state object and pass it to the drm_atomic_helper_resume() helper upon
/// resume.
///
/// See also:
/// drm_atomic_helper_duplicate_state(), drm_atomic_helper_disable_all(),
/// drm_atomic_helper_resume(), drm_atomic_helper_commit_duplicated_state()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_suspend")]
extern "C" fn drm_atomic_helper_suspend(dev: *mut drm_device) -> *mut drm_atomic_commit {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
let mut ctx: core::mem::MaybeUninit<drm_modeset_acquire_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut drm_modeset_acquire_ctx = core::ptr::addr_of_mut!(ctx) as *mut drm_modeset_acquire_ctx;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// This can never be returned, but it makes the compiler happy
*__slate_slot_1 = ERR_PTR(22i32.wrapping_neg() as i64) as *mut drm_atomic_commit;
if !drm_drv_uses_atomic_modeset(dev) {
mutex_lock(core::ptr::addr_of_mut!((*dev).mode_config.mutex));
}
drm_modeset_acquire_init(core::ptr::addr_of_mut!(*__slate_slot_2), 0u32);
loop {
*__slate_slot_0 = drm_modeset_lock_all_ctx(dev, core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_0 == 0 {
*__slate_slot_1 = drm_atomic_helper_duplicate_state(dev, core::ptr::addr_of_mut!(*__slate_slot_2));
if !IS_ERR(*__slate_slot_1 as *const ()) {
*__slate_slot_0 = drm_atomic_helper_disable_all(dev, core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_0 < 0 {
drm_atomic_commit_put(*__slate_slot_1);
*__slate_slot_1 = ERR_PTR(*__slate_slot_0 as i64) as *mut drm_atomic_commit;
}
}
}
if *__slate_slot_0 == 35i32.wrapping_neg() {
*__slate_slot_0 = drm_modeset_backoff(core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_0 != 0 {
break;
}
} else {
break;
}
}
drm_modeset_drop_locks(core::ptr::addr_of_mut!(*__slate_slot_2));
drm_modeset_acquire_fini(core::ptr::addr_of_mut!(*__slate_slot_2));
if !drm_drv_uses_atomic_modeset(dev) {
mutex_unlock(core::ptr::addr_of_mut!((*dev).mode_config.mutex));
}
if *__slate_slot_0 != 0 {
return ERR_PTR(*__slate_slot_0 as i64) as *mut drm_atomic_commit;
} else {
return *__slate_slot_1;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_suspend_680: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device) -> *mut drm_atomic_commit>, *mut ()>(Some(drm_atomic_helper_suspend)) };

/// drm_atomic_helper_commit_duplicated_state - commit duplicated state
/// @state: duplicated atomic state to commit
/// @ctx: pointer to acquire_ctx to use for commit.
///
/// The state returned by drm_atomic_helper_duplicate_state() and
/// drm_atomic_helper_suspend() is partially invalid, and needs to
/// be fixed up before commit.
///
/// Returns:
/// 0 on success or a negative error code on failure.
///
/// See also:
/// drm_atomic_helper_suspend()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_commit_duplicated_state")]
extern "C" fn drm_atomic_helper_commit_duplicated_state(state: *mut drm_atomic_commit, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut i: i32 = 0;
let mut ret: i32 = 0;
let mut plane: *mut drm_plane = unsafe { core::mem::zeroed() };
let mut colorop: *mut drm_colorop = unsafe { core::mem::zeroed() };
let mut connector: *mut drm_connector = unsafe { core::mem::zeroed() };
let mut crtc: *mut drm_crtc = unsafe { core::mem::zeroed() };
unsafe {
(*state).acquire_ctx = ctx;
}
while i < unsafe { (*(*state).dev).mode_config.num_colorop } {
let __v0: bool = if !unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).ptr }.is_null() {
colorop = unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).ptr };
let new_colorop_state: *mut drm_colorop_state = unsafe { (*unsafe { (*state).colorops }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v0 {
unsafe {
(*unsafe { (*state).colorops }.wrapping_offset(i as isize)).old_state = (*colorop).state;
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_total_plane } {
let __v5: bool = if !unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr }.is_null() {
plane = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).ptr };
let new_plane_state: *mut drm_plane_state = unsafe { (*unsafe { (*state).planes }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v5 {
unsafe {
(*unsafe { (*state).planes }.wrapping_offset(i as isize)).old_state = (*plane).state;
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*(*state).dev).mode_config.num_crtc } {
let __v10: bool = if !unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr }.is_null() {
crtc = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).ptr };
let new_crtc_state: *mut drm_crtc_state = unsafe { (*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v10 {
unsafe {
(*unsafe { (*state).crtcs }.wrapping_offset(i as isize)).old_state = (*crtc).state;
}
}
i = i.wrapping_add(1);
}
i = 0;
while i < unsafe { (*state).num_connector } {
let __v15: bool = if !unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr }.is_null() {
connector = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).ptr };
let new_conn_state: *mut drm_connector_state = unsafe { (*unsafe { (*state).connectors }.wrapping_offset(i as isize)).new_state };
true
} else {
false
};
if __v15 {
unsafe {
(*unsafe { (*state).connectors }.wrapping_offset(i as isize)).old_state = (*connector).state;
}
}
i = i.wrapping_add(1);
}
ret = unsafe { drm_atomic_commit(state) };
unsafe {
(*state).acquire_ctx = core::ptr::null_mut::<drm_modeset_acquire_ctx>();
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_commit_duplicated_state_681: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_atomic_commit, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_commit_duplicated_state)) };

/// drm_atomic_helper_resume - subsystem-level resume helper
/// @dev: DRM device
/// @state: atomic state to resume to
///
/// Calls drm_mode_config_reset() to synchronize hardware and software states,
/// grabs all modeset locks and commits the atomic state object. This can be
/// used in conjunction with the drm_atomic_helper_suspend() helper to
/// implement suspend/resume for drivers that support atomic mode-setting.
///
/// Returns:
/// 0 on success or a negative error code on failure.
///
/// See also:
/// drm_atomic_helper_suspend()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_resume")]
extern "C" fn drm_atomic_helper_resume(dev: *mut drm_device, state: *mut drm_atomic_commit) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut ctx: core::mem::MaybeUninit<drm_modeset_acquire_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut drm_modeset_acquire_ctx = core::ptr::addr_of_mut!(ctx) as *mut drm_modeset_acquire_ctx;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
drm_mode_config_reset(dev);
if !drm_drv_uses_atomic_modeset(dev) {
mutex_lock(core::ptr::addr_of_mut!((*dev).mode_config.mutex));
}
drm_modeset_acquire_init(core::ptr::addr_of_mut!(*__slate_slot_1), 0u32);
loop {
*__slate_slot_0 = drm_modeset_lock_all_ctx(dev, core::ptr::addr_of_mut!(*__slate_slot_1));
if *__slate_slot_0 == 0 {
*__slate_slot_0 = drm_atomic_helper_commit_duplicated_state(state, core::ptr::addr_of_mut!(*__slate_slot_1));
}
if *__slate_slot_0 == 35i32.wrapping_neg() {
*__slate_slot_0 = drm_modeset_backoff(core::ptr::addr_of_mut!(*__slate_slot_1));
if *__slate_slot_0 != 0 {
break;
}
} else {
break;
}
}
drm_modeset_drop_locks(core::ptr::addr_of_mut!(*__slate_slot_1));
drm_modeset_acquire_fini(core::ptr::addr_of_mut!(*__slate_slot_1));
if !drm_drv_uses_atomic_modeset(dev) {
mutex_unlock(core::ptr::addr_of_mut!((*dev).mode_config.mutex));
}
drm_atomic_commit_put(state);
return *__slate_slot_0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_resume_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>, *mut ()>(Some(drm_atomic_helper_resume)) };

fn page_flip_common(state: *mut drm_atomic_commit, crtc: *mut drm_crtc, fb: *mut drm_framebuffer, event: *mut drm_pending_vblank_event, flags: u32) -> i32 {
let plane: *mut drm_plane = unsafe { (*crtc).primary };
let mut ret: i32 = 0;
let crtc_state: *mut drm_crtc_state = unsafe { drm_atomic_get_crtc_state(state, crtc) };
if IS_ERR(crtc_state as *const ()) {
return PTR_ERR(crtc_state as *const ()) as i32;
}
unsafe {
(*crtc_state).event = event;
}
unsafe {
(*crtc_state).async_flip = flags & 2 != 0;
}
let plane_state: *mut drm_plane_state = unsafe { drm_atomic_get_plane_state(state, plane) };
if IS_ERR(plane_state as *const ()) {
return PTR_ERR(plane_state as *const ()) as i32;
}
ret = unsafe { drm_atomic_set_crtc_for_plane(plane_state, crtc) };
if ret != 0 {
return ret;
}
unsafe { drm_atomic_set_fb_for_plane(plane_state, fb) };
// Make sure we don't accidentally do a full modeset.
unsafe {
let __t0 = false;
(*state).__slate_bits_0.__set_allow_modeset(__t0);
}
if !unsafe { (*crtc_state).active } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*crtc).dev as *const drm_device) as *const device, 4u32, c"[CRTC:%d:%s] disabled, rejecting legacy flip\n".as_ptr() as *const u8, (*crtc).base.id, (*crtc).name) };
return 22i32.wrapping_neg();
}
ret
}

/// drm_atomic_helper_page_flip - execute a legacy page flip
/// @crtc: DRM CRTC
/// @fb: DRM framebuffer
/// @event: optional DRM event to signal upon completion
/// @flags: flip flags for non-vblank sync'ed updates
/// @ctx: lock acquisition context
///
/// Provides a default &drm_crtc_funcs.page_flip implementation
/// using the atomic driver interface.
///
/// Returns:
/// Returns 0 on success, negative errno numbers on failure.
///
/// See also:
/// drm_atomic_helper_page_flip_target()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_page_flip")]
extern "C" fn drm_atomic_helper_page_flip(crtc: *mut drm_crtc, fb: *mut drm_framebuffer, event: *mut drm_pending_vblank_event, flags: u32, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let plane: *mut drm_plane = (*crtc).primary;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_1 = drm_atomic_commit_alloc((*plane).dev);
if (*__slate_slot_1).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_1)).acquire_ctx = ctx;
ret = page_flip_common(*__slate_slot_1, crtc, fb, event, flags);
if ret == 0 {
ret = drm_atomic_nonblocking_commit(*__slate_slot_1);
}
drm_atomic_commit_put(*__slate_slot_1);
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_page_flip_683: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_page_flip)) };

/// drm_atomic_helper_page_flip_target - do page flip on target vblank period.
/// @crtc: DRM CRTC
/// @fb: DRM framebuffer
/// @event: optional DRM event to signal upon completion
/// @flags: flip flags for non-vblank sync'ed updates
/// @target: specifying the target vblank period when the flip to take effect
/// @ctx: lock acquisition context
///
/// Provides a default &drm_crtc_funcs.page_flip_target implementation.
/// Similar to drm_atomic_helper_page_flip() with extra parameter to specify
/// target vblank period to flip.
///
/// Returns:
/// Returns 0 on success, negative errno numbers on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_page_flip_target")]
extern "C" fn drm_atomic_helper_page_flip_target(crtc: *mut drm_crtc, fb: *mut drm_framebuffer, event: *mut drm_pending_vblank_event, flags: u32, target: u32, ctx: *mut drm_modeset_acquire_ctx) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut crtc_state: core::mem::MaybeUninit<*mut drm_crtc_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut drm_crtc_state = core::ptr::addr_of_mut!(crtc_state) as *mut *mut drm_crtc_state;
let mut state: core::mem::MaybeUninit<*mut drm_atomic_commit> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut drm_atomic_commit = core::ptr::addr_of_mut!(state) as *mut *mut drm_atomic_commit;
unsafe {
let plane: *mut drm_plane = (*crtc).primary;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_5 = drm_atomic_commit_alloc((*plane).dev);
if (*__slate_slot_5).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_5)).acquire_ctx = ctx;
ret = page_flip_common(*__slate_slot_5, crtc, fb, event, flags);
if ret == 0 {
*__slate_slot_4 = drm_atomic_get_new_crtc_state(*__slate_slot_5 as *const drm_atomic_commit, crtc);
__ret_warn_on = (*__slate_slot_4).is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("684: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 684b - ., 3; .popsection\n# {0}", const 684 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_atomic_helper_page_flip_target_7, sym __slate_asm_string_drm_atomic_helper_page_flip_target_8, const 4058 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("685: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 685b - ., 4; .popsection\n# {0}", const 685 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
ret = 22i32.wrapping_neg();
} else {
(*(*__slate_slot_4)).target_vblank = target;
ret = drm_atomic_nonblocking_commit(*__slate_slot_5);
}
}
drm_atomic_commit_put(*__slate_slot_5);
return ret;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_page_flip_target_686: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>, *mut ()>(Some(drm_atomic_helper_page_flip_target)) };

/// drm_atomic_helper_bridge_propagate_bus_fmt() - Propagate output format to
/// 						  the input end of a bridge
/// @bridge: bridge control structure
/// @bridge_state: new bridge state
/// @crtc_state: new CRTC state
/// @conn_state: new connector state
/// @output_fmt: tested output bus format
/// @num_input_fmts: will contain the size of the returned array
///
/// This helper is a pluggable implementation of the
/// &drm_bridge_funcs.atomic_get_input_bus_fmts operation for bridges that don't
/// modify the bus configuration between their input and their output. It
/// returns an array of input formats with a single element set to @output_fmt.
///
/// RETURNS:
/// a valid format array of size @num_input_fmts, or NULL if the allocation
/// failed
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_atomic_helper.drm_atomic_helper_bridge_propagate_bus_fmt")]
extern "C" fn drm_atomic_helper_bridge_propagate_bus_fmt(bridge: *mut drm_bridge, bridge_state: *mut drm_bridge_state, crtc_state: *mut drm_crtc_state, conn_state: *mut drm_connector_state, output_fmt: u32, num_input_fmts: *mut u32) -> *mut u32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut u32;
let __obj_size: u64 = size_mul(core::mem::size_of::<u32>() as u64, 1u64);
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
__v0 = __v1 as *mut u32;
let input_fmts: *mut u32 = __v0;
if input_fmts.is_null() {
unsafe {
*num_input_fmts = 0;
}
return core::ptr::null_mut::<u32>();
}
unsafe {
*num_input_fmts = 1;
}
unsafe {
*input_fmts.wrapping_offset(0 as isize) = output_fmt;
}
input_fmts
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_bridge_propagate_bus_fmt_687: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_bridge_state, *mut drm_crtc_state, *mut drm_connector_state, u32, *mut u32) -> *mut u32>, *mut ()>(Some(drm_atomic_helper_bridge_propagate_bus_fmt)) };

// drm_atomic_helper_bridge_get_hdmi_output_bus_fmts - helper implementing
//                                           atomic_get_output_bus_fmts for HDMI
// @bridge: pointer to &struct drm_bridge
// @bridge_state: pointer to the current bridge state
// @crtc_state: pointer to the current CRTC state
// @conn_state: pointer to the current connector state
// @num_output_fmts: pointer to where the number of entries in the returned array
//                   will be stored. Set to 0 if unsuccessful.
//
// Common implementation for the &drm_bridge_funcs.atomic_get_output_bus_fmts
// operation that's applicable to HDMI connectors.
//
// Returns: a newly allocated array of u32 values of length \*@num_output_fmts,
// representing all the MEDIA_BUS_FMTS\_ for the current connector state's
// chosen HDMI output bits per compoennt, or %NULL if it fails to allocate one.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_atomic_helper_bridge_get_hdmi_output_bus_fmts_689: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_bridge, *mut drm_bridge_state, *mut drm_crtc_state, *mut drm_connector_state, *mut u32) -> *mut u32>, *mut ()>(Some(drm_atomic_helper_bridge_get_hdmi_output_bus_fmts)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_drm_atomic_helper_async_commit_13: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_async_commit_14: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_async_commit_18: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_async_commit_19: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_async_commit_23: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_async_commit_24: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_async_commit_28: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_async_commit_29: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_async_commit_3: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_async_commit_4: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_async_commit_8: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_async_commit_9: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_check_modeset_12: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_check_modeset_13: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_check_modeset_23: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_check_modeset_24: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_check_plane_state_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_check_plane_state_2: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_check_plane_state_6: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_check_plane_state_7: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_check_planes_4: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_check_planes_5: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_cleanup_done_10: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_cleanup_done_16: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_cleanup_done_17: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_cleanup_done_3: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_cleanup_done_4: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_cleanup_done_9: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_crtc_disable_13: [u8; 45] = *b"driver forgot to call drm_crtc_vblank_off()\n\0";

static __slate_asm_string_drm_atomic_helper_commit_crtc_disable_14: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_crtc_disable_7: [u8; 40] = *b"driver disabled vblank in self-refresh\n\0";

static __slate_asm_string_drm_atomic_helper_commit_crtc_disable_8: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_disable_6: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_disable_7: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_post_disable_6: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_encoder_bridge_post_disable_7: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_hw_done_5: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_hw_done_6: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_planes_on_crtc_5: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_planes_on_crtc_6: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_commit_writebacks_4: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_commit_writebacks_5: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_disable_planes_on_crtc_3: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_disable_planes_on_crtc_4: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_page_flip_target_7: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_page_flip_target_8: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_plane_changed_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_plane_changed_2: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_plane_changed_6: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_plane_changed_7: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_prepare_planes_34: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_prepare_planes_35: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_swap_state_19: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_swap_state_20: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_swap_state_30: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_swap_state_31: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_swap_state_41: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_swap_state_42: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_swap_state_52: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_swap_state_53: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_swap_state_60: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_swap_state_61: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_17: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_18: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_4: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_update_legacy_modeset_state_5: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_wait_for_fences_4: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_helper_wait_for_fences_5: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_helper_wait_for_vblanks_35: [u8; 36] = *b"[CRTC:%d:%s] vblank wait timed out\n\0";

static __slate_asm_string_drm_atomic_helper_wait_for_vblanks_36: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_drm_atomic_plane_disabling_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_plane_disabling_2: [u8; 61] = *b"/home/takashi/c-corpus/linux/include/drm/drm_atomic_helper.h\0";

static __slate_asm_string_drm_atomic_plane_enabling_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_atomic_plane_enabling_2: [u8; 61] = *b"/home/takashi/c-corpus/linux/include/drm/drm_atomic_helper.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_ktime_divns_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_divns_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/include/linux/ktime.h\0";

static __slate_asm_string_mode_fixup_8: [u8; 1] = *b"\0";

static __slate_asm_string_mode_fixup_9: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_set_best_encoder_1: [u8; 1] = *b"\0";

static __slate_asm_string_set_best_encoder_10: [u8; 1] = *b"\0";

static __slate_asm_string_set_best_encoder_11: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";

static __slate_asm_string_set_best_encoder_2: [u8; 65] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_atomic_helper.c\0";
