
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
//! Copyright © 2018 Intel Corporation
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sublicense,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice (including the next
//! paragraph) shall be included in all copies or substantial portions of the
//! Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
//! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//! DEALINGS IN THE SOFTWARE.
//!
//! Authors:
//!   Madhav Chauhan <madhav.chauhan@intel.com>
//!   Jani Nikula <jani.nikula@intel.com>
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
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! # chars in a path name including nul
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! This counts to 15. Any more, it will return 16th argument.
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
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
//! New user namespace
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
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! Let people know that we have them
//! See the comment in asm-generic/io.h about ioremap_np().
//! poll_timeout_us - Periodically poll and perform an operation until
//!                   a condition is met or a timeout occurs
//!
//! @op: Operation
//! @cond: Break condition
//! @sleep_us: Maximum time to sleep between operations in us (0 tight-loops).
//!            Please read usleep_range() function description for details and
//!            limitations.
//! @timeout_us: Timeout in us, 0 means never timeout
//! @sleep_before_op: if it is true, sleep @sleep_us before operation.
//!
//! When available, you'll probably want to use one of the specialized
//! macros defined below rather than this macro directly.
//!
//! Returns: 0 on success and -ETIMEDOUT upon a timeout. Must not
//! be called from atomic context if sleep_us or timeout_us are used.
//! Receiver Capability
//! VESA Display Stream Compression DSC 1.2 constants
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
//! Generic info of form tag = "info"
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
//!  + ((B + 0.000122)*4096))
//! use Low Power Mode to transmit message
//! Gen11 DSI
//! Utility pin
//! ICL Port CL_DW registers
//! Display platforms and subplatforms. Keep platforms in display version based
//! order, chronological order within a version, and subplatforms next to the
//! platform.
//! Invoke param, a function-like macro, for each intel display param, with
//! arguments:
//!
//! param(type, name, value, mode)
//!
//! type: parameter type, one of {bool, int, unsigned int, unsigned long, char *}
//! name: name of the parameter
//! value: initial/default value of the parameter
//! mode: debugfs file permissions, one of {0400, 0600, 0}, use 0 to not create
//!       debugfs file
//! -ENOENT means we got the ref, but there's no tracking
//! including 0
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! Amount of SAGV/QGV points, BSpec precisely defines this
//! i965+
//! Helpful configuration macro to be used in tables etc.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
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
//! DWORD 3
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! DRM_FORMAT_MAX_PLANES - maximum number of planes a DRM format can have
//! Following Macros take a byte at a time for bit(s) masking
//!
//! TODO: HDCP_2_2_MAX_CONTENT_STREAMS_CNT is based upon actual
//! H/W MST streams capacity.
//! This required to be moved out to platform specific header.
//! Each region is a minimum of 16k, and there are at most 255 of them.
//! Display Stream Splitter Control
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
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// "apic" Onboard APIC
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Xen paravirtual guest
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
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
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
// Overflow Flag
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
// Protocol error
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// SCHED_ISO: reserved but not implemented yet
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// guarantee 'op' and 'cond' are evaluated after timeout expired
// 1.3
// 1.2
// 2.0
// eDP
// XXX 1.2?
// 16-bit little-endian
// DP 1.4
// eDP 1.4
// eDP 1.4
// DP-HDMI2.1 PCON DSC ENCODER SUPPORT
// 0x92 through 0x9E
// DSC Capabilities 0x60 through 0x6F
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
// Given the arbitrary numbers in varargs, pick the 0-based __index'th number.
//
// Always prefer _PICK_EVEN() over this if the numbers are evenly spaced.
// Local integer constant expression version of is_power_of_2().
// REG_FIELD_PREP() - Prepare a u32 bitfield value
// @__mask: shifted mask defining the field's length and position
// @__val: value to put in the field
//
// Local copy of FIELD_PREP() to generate an integer constant expression, force
// u32 and for consistency with REG_FIELD_GET(), REG_BIT() and REG_GENMASK().
//
// @return: @__val masked and shifted into the field defined by @__mask.
// in KHz
// tgl+
// tgl+
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
// Known as DDI_CTL_DE in MTL+
// ICL Clocks
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Fixed by ACPI spec
// DWORD 5
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
// table size 2k - maximum due to use
// of chars for next/prev indices
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
macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8)
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1258>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__27 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_263_17>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__35 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1057>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__72 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1326>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__89 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_450>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__155 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_581>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__158 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_494>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__223 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1053>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__237 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1259>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__375 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1403>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__386 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_263_18>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__407 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_534>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__564 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_453_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__702 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_453>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__734 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1400>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__758 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_537>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__775 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1322>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__814 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1759>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__871 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1610>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__876 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_817>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__889 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_492>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__982 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_522>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__992 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_520>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1023 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_271_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1060 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_509>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1339 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_511>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1459 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1386>() as i32
};
}

macro_rules! BIT_TYPE__87 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__407!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__98 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__889!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__144 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__992!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__216 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__758!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__255 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1339!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__350 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__876!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__436 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__237!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__529 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__386!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__555 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__871!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__642 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__702!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__690 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__72!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__746 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__27!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__796 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__564!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__987 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__375!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1032 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__982!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1210 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1023!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1250 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__734!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1277 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__814!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1292 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1302 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__35!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1305 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1060!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1309 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__155!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1362 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__223!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1414 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__775!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1445 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__158!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1489 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__89!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1518 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1459!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_U32__28 {
($nr:expr) => {
BIT_TYPE__87!(u32, $nr)
};
}

macro_rules! BIT_U32__149 {
($nr:expr) => {
BIT_TYPE__144!(u32, $nr)
};
}

macro_rules! BIT_U32__194 {
($nr:expr) => {
BIT_TYPE__1032!(u32, $nr)
};
}

macro_rules! BIT_U32__227 {
($nr:expr) => {
BIT_TYPE__1309!(u32, $nr)
};
}

macro_rules! BIT_U32__294 {
($nr:expr) => {
BIT_TYPE__690!(u32, $nr)
};
}

macro_rules! BIT_U32__295 {
($nr:expr) => {
BIT_TYPE__1362!(u32, $nr)
};
}

macro_rules! BIT_U32__322 {
($nr:expr) => {
BIT_TYPE__529!(u32, $nr)
};
}

macro_rules! BIT_U32__327 {
($nr:expr) => {
BIT_TYPE__796!(u32, $nr)
};
}

macro_rules! BIT_U32__341 {
($nr:expr) => {
BIT_TYPE__1210!(u32, $nr)
};
}

macro_rules! BIT_U32__485 {
($nr:expr) => {
BIT_TYPE__746!(u32, $nr)
};
}

macro_rules! BIT_U32__730 {
($nr:expr) => {
BIT_TYPE__1250!(u32, $nr)
};
}

macro_rules! BIT_U32__832 {
($nr:expr) => {
BIT_TYPE__350!(u32, $nr)
};
}

macro_rules! BIT_U32__894 {
($nr:expr) => {
BIT_TYPE__555!(u32, $nr)
};
}

macro_rules! BIT_U32__919 {
($nr:expr) => {
BIT_TYPE__1414!(u32, $nr)
};
}

macro_rules! BIT_U32__973 {
($nr:expr) => {
BIT_TYPE__1277!(u32, $nr)
};
}

macro_rules! BIT_U32__1007 {
($nr:expr) => {
BIT_TYPE__1302!(u32, $nr)
};
}

macro_rules! BIT_U32__1017 {
($nr:expr) => {
BIT_TYPE__1292!(u32, $nr)
};
}

macro_rules! BIT_U32__1067 {
($nr:expr) => {
BIT_TYPE__987!(u32, $nr)
};
}

macro_rules! BIT_U32__1081 {
($nr:expr) => {
BIT_TYPE__255!(u32, $nr)
};
}

macro_rules! BIT_U32__1171 {
($nr:expr) => {
BIT_TYPE__1445!(u32, $nr)
};
}

macro_rules! BIT_U32__1216 {
($nr:expr) => {
BIT_TYPE__98!(u32, $nr)
};
}

macro_rules! BIT_U32__1237 {
($nr:expr) => {
BIT_TYPE__436!(u32, $nr)
};
}

macro_rules! BIT_U32__1278 {
($nr:expr) => {
BIT_TYPE__1305!(u32, $nr)
};
}

macro_rules! BIT_U32__1297 {
($nr:expr) => {
BIT_TYPE__216!(u32, $nr)
};
}

macro_rules! BIT_U32__1353 {
($nr:expr) => {
BIT_TYPE__1518!(u32, $nr)
};
}

macro_rules! BIT_U32__1392 {
($nr:expr) => {
BIT_TYPE__1489!(u32, $nr)
};
}

macro_rules! BIT_U32__1422 {
($nr:expr) => {
BIT_TYPE__642!(u32, $nr)
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BIT__2 {
($nr:expr) => {
UL!() << (($nr) as i32)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! CLK_PREPARE {
($x:expr) => {
($x) << 28i32
};
}

macro_rules! CLK_PRE {
($x:expr) => {
($x) << 16i32
};
}

macro_rules! CLK_ZERO {
($x:expr) => {
($x) << 20i32
};
}

macro_rules! CURSOR_COEFF {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_284_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_284_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_284_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_284_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_284_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_284_16>() as u32)
};
}

macro_rules! DISPLAY_INFO {
($__display:expr) => {
unsafe { (*$__display).info.__device_info }
};
}

macro_rules! DISPLAY_MMIO_BASE {
($display:expr) => {
unsafe { (*DISPLAY_INFO!($display)).mmio_offset }
};
}

macro_rules! DISPLAY_RUNTIME_INFO {
($__display:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__display).info.__runtime_info) }
};
}

macro_rules! DISPLAY_VER {
($__display:expr) => {
unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).ip.ver }
};
}

macro_rules! FRC_LATENCY_OPTIM_VAL {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_459_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 8i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(10)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_459_4>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 8i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(10))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_459_5>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_459_7>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_459_10>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_459_17>() as u32)
};
}

macro_rules! FRC_LATENCY_OPTIM_VAL__2 {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_462_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 8i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(10)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_462_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 8i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(10))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_462_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_462_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_462_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_462_16>() as u32)
};
}

macro_rules! GENMASK_INPUT_CHECK__36 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_282>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__47 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_461>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__72 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_271>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__79 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_473>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__147 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_281_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__149 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_262>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__166 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_596>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__204 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_1277>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__212 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_830>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__221 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_469>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__230 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_281>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__232 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_272>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__282 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_262_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__284 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_459>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__356 {
() => {
core::mem::size_of::<__SlateRecord_icl_dsi_c_504>() as i32
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__19 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__147!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__48 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__36!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__51 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__282!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__78 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__149!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__101 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__221!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__122 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__230!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__125 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__356!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__150 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__284!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__157 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__204!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__171 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__166!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__175 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__47!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__213 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__212!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__294 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__232!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__333 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__72!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__349 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__79!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
};
}

macro_rules! GENMASK_U32 {
($h:expr, $l:expr) => {
GENMASK_TYPE__101!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__25 {
($h:expr, $l:expr) => {
GENMASK_TYPE__213!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__31 {
($h:expr, $l:expr) => {
GENMASK_TYPE__175!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__39 {
($h:expr, $l:expr) => {
GENMASK_TYPE__150!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__69 {
($h:expr, $l:expr) => {
GENMASK_TYPE__294!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__81 {
($h:expr, $l:expr) => {
GENMASK_TYPE__51!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__91 {
($h:expr, $l:expr) => {
GENMASK_TYPE__48!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__106 {
($h:expr, $l:expr) => {
GENMASK_TYPE__122!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__109 {
($h:expr, $l:expr) => {
GENMASK_TYPE__19!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__167 {
($h:expr, $l:expr) => {
GENMASK_TYPE__349!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__256 {
($h:expr, $l:expr) => {
GENMASK_TYPE__333!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__277 {
($h:expr, $l:expr) => {
GENMASK_TYPE__157!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__280 {
($h:expr, $l:expr) => {
GENMASK_TYPE__78!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__282 {
($h:expr, $l:expr) => {
GENMASK_TYPE__125!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__283 {
($h:expr, $l:expr) => {
GENMASK_TYPE__171!(u32, $h, $l)
};
}

macro_rules! HACTIVE__2 {
($hdisplay:expr) => {
((($hdisplay) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_955_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_955_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_16>() as u32)
};
}

macro_rules! HSTX_TIMEOUT_VALUE {
($x:expr) => {
($x) << 16i32
};
}

macro_rules! HSYNC_END__2 {
($hsync_end:expr) => {
((($hsync_end) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_979_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_979_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_32>() as u32)
};
}

macro_rules! HSYNC_START__3 {
($hsync_start:expr) => {
((($hsync_start) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_979_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_979_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_979_16>() as u32)
};
}

macro_rules! HS_EXIT {
($x:expr) => {
($x) << 0i32
};
}

macro_rules! HS_PREPARE {
($x:expr) => {
($x) << 24i32
};
}

macro_rules! HS_ZERO {
($x:expr) => {
($x) << 16i32
};
}

macro_rules! HTOTAL {
($htotal:expr) => {
((($htotal) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_955_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_955_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_955_32>() as u32)
};
}

macro_rules! ICL_DPCLKA_CFGCR0_DDI_CLK_SEL_MASK {
($phy:expr) => {
3i32 << ICL_DPCLKA_CFGCR0_DDI_CLK_SEL_SHIFT!($phy)
};
}

macro_rules! ICL_DPCLKA_CFGCR0_DDI_CLK_SEL_SHIFT {
($phy:expr) => {
(($phy) as i32).wrapping_mul(2)
};
}

macro_rules! ICL_DPCLKA_CFGCR0_DDI_CLK_SEL {
($pll:expr, $phy:expr) => {
(($pll) as i32) << ICL_DPCLKA_CFGCR0_DDI_CLK_SEL_SHIFT!($phy)
};
}

macro_rules! INTEL_DISPLAY_DEVICE_PIPE_OFFSET {
($display:expr, $pipe:expr) => {
unsafe { *((*DISPLAY_INFO!($display)).pipe_offsets.as_ptr() as *const u32).wrapping_offset((($pipe) as i32) as isize) }.wrapping_sub(unsafe { *((*DISPLAY_INFO!($display)).pipe_offsets.as_ptr() as *const u32).wrapping_offset(0 as isize) }).wrapping_add(DISPLAY_MMIO_BASE!($display))
};
}

macro_rules! INTEL_DISPLAY_DEVICE_TRANS_OFFSET {
($display:expr, $trans:expr) => {
unsafe { *((*DISPLAY_INFO!($display)).trans_offsets.as_ptr() as *const u32).wrapping_offset((($trans) as i32) as isize) }.wrapping_sub(unsafe { *((*DISPLAY_INFO!($display)).trans_offsets.as_ptr() as *const u32).wrapping_offset(0 as isize) }).wrapping_add(DISPLAY_MMIO_BASE!($display))
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

macro_rules! LATENCY_OPTIM_VAL {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_474_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 2i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_474_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 2i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_474_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_474_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_474_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_474_16>() as u32)
};
}

macro_rules! LATENCY_OPTIM_VAL__2 {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_469_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 2i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_469_4>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 2i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_469_5>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_469_7>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_469_10>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_469_17>() as u32)
};
}

macro_rules! LEFT_DL_BUF_TARGET_DEPTH {
($pixels:expr) => {
(($pixels) as i32) << 0i32
};
}

macro_rules! LPRX_TIMEOUT_VALUE {
($x:expr) => {
($x) << 0i32
};
}

macro_rules! OVERLAP_PIXELS {
($pixels:expr) => {
(($pixels) as i32) << 16i32
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PIX_VIRT_CHAN {
($x:expr) => {
($x) << 12i32
};
}

macro_rules! POST_CURSOR_1__2 {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_283_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(17)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_283_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(17))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_16>() as u32)
};
}

macro_rules! POST_CURSOR_2 {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_283_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 6i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_283_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 6i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_283_32>() as u32)
};
}

macro_rules! RCOMP_SCALAR__2 {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_274_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_274_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_274_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_274_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_274_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_274_16>() as u32)
};
}

macro_rules! REG_BIT__127 {
($n:expr) => {
BIT_U32__1067!($n)
};
}

macro_rules! REG_BIT__187 {
($n:expr) => {
BIT_U32__149!($n)
};
}

macro_rules! REG_BIT__264 {
($n:expr) => {
BIT_U32__341!($n)
};
}

macro_rules! REG_BIT__422 {
($n:expr) => {
BIT_U32__1422!($n)
};
}

macro_rules! REG_BIT__455 {
($n:expr) => {
BIT_U32__919!($n)
};
}

macro_rules! REG_BIT__462 {
($n:expr) => {
BIT_U32__295!($n)
};
}

macro_rules! REG_BIT__539 {
($n:expr) => {
BIT_U32__194!($n)
};
}

macro_rules! REG_BIT__583 {
($n:expr) => {
BIT_U32__227!($n)
};
}

macro_rules! REG_BIT__682 {
($n:expr) => {
BIT_U32__894!($n)
};
}

macro_rules! REG_BIT__757 {
($n:expr) => {
BIT_U32__1353!($n)
};
}

macro_rules! REG_BIT__780 {
($n:expr) => {
BIT_U32__294!($n)
};
}

macro_rules! REG_BIT__834 {
($n:expr) => {
BIT_U32__1216!($n)
};
}

macro_rules! REG_BIT__897 {
($n:expr) => {
BIT_U32__485!($n)
};
}

macro_rules! REG_BIT__943 {
($n:expr) => {
BIT_U32__1392!($n)
};
}

macro_rules! REG_BIT__1002 {
($n:expr) => {
BIT_U32__1278!($n)
};
}

macro_rules! REG_BIT__1032 {
($n:expr) => {
BIT_U32__1007!($n)
};
}

macro_rules! REG_BIT__1052 {
($n:expr) => {
BIT_U32__28!($n)
};
}

macro_rules! REG_BIT__1076 {
($n:expr) => {
BIT_U32__832!($n)
};
}

macro_rules! REG_BIT__1161 {
($n:expr) => {
BIT_U32__322!($n)
};
}

macro_rules! REG_BIT__1200 {
($n:expr) => {
BIT_U32__1017!($n)
};
}

macro_rules! REG_BIT__1264 {
($n:expr) => {
BIT_U32__1171!($n)
};
}

macro_rules! REG_BIT__1289 {
($n:expr) => {
BIT_U32__1081!($n)
};
}

macro_rules! REG_BIT__1317 {
($n:expr) => {
BIT_U32__1237!($n)
};
}

macro_rules! REG_BIT__1373 {
($n:expr) => {
BIT_U32__1297!($n)
};
}

macro_rules! REG_BIT__1430 {
($n:expr) => {
BIT_U32__327!($n)
};
}

macro_rules! REG_BIT__1441 {
($n:expr) => {
BIT_U32__730!($n)
};
}

macro_rules! REG_BIT__1449 {
($n:expr) => {
BIT_U32__973!($n)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__24 {
($high:expr, $low:expr) => {
GENMASK_U32!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__44 {
($high:expr, $low:expr) => {
GENMASK_U32__91!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__56 {
($high:expr, $low:expr) => {
GENMASK_U32__283!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__67 {
($high:expr, $low:expr) => {
GENMASK_U32__277!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__70 {
($high:expr, $low:expr) => {
GENMASK_U32__31!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__85 {
($high:expr, $low:expr) => {
GENMASK_U32__167!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__97 {
($high:expr, $low:expr) => {
GENMASK_U32__25!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__114 {
($high:expr, $low:expr) => {
GENMASK_U32__106!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__137 {
($high:expr, $low:expr) => {
GENMASK_U32__39!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__157 {
($high:expr, $low:expr) => {
GENMASK_U32__69!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__163 {
($high:expr, $low:expr) => {
GENMASK_U32__282!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__164 {
($high:expr, $low:expr) => {
GENMASK_U32__280!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__211 {
($high:expr, $low:expr) => {
GENMASK_U32__81!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__230 {
($high:expr, $low:expr) => {
GENMASK_U32__109!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__261 {
($high:expr, $low:expr) => {
GENMASK_U32__256!($high, $low)
};
}

macro_rules! RIGHT_DL_BUF_TARGET_DEPTH {
($pixels:expr) => {
(($pixels) as i32) << 0i32
};
}

macro_rules! RTERM_SELECT {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_264_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 3i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_264_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 3i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_264_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_264_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_264_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_264_16>() as u32)
};
}

macro_rules! SCALING_MODE_SEL {
($x:expr) => {
((($x) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_263_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 18i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(20)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_263_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 18i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(20))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_263_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_263_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_263_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_263_16>() as u32)
};
}

macro_rules! SWING_SEL_LOWER {
($x:expr) => {
(((($x) & 7) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_273_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 11i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_273_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 11i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(13))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_32>() as u32)
};
}

macro_rules! SWING_SEL_UPPER {
($x:expr) => {
(((($x) >> 3i32) as u32) << ((((core::mem::size_of::<__SlateRecord_icl_dsi_c_273_2>() as u64).wrapping_add(32768u64) as u32) as u64).trailing_zeros() as i32) & ((core::mem::size_of::<__SlateRecord_icl_dsi_c_273_3>() as u64).wrapping_add(32768u64) as u32)).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_273_16>() as u32)
};
}

macro_rules! TA_SURE {
($x:expr) => {
($x) << 16i32
};
}

macro_rules! TA_TIMEOUT_VALUE {
($x:expr) => {
($x) << 0i32
};
}

macro_rules! TGL_DSI_CHKN_LSHS_GB {
($byte_clocks:expr) => {
((($byte_clocks) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_1278_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_1278_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1278_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1278_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1278_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1278_16>() as u32)
};
}

macro_rules! TRANS_DDI_PORT_WIDTH__2 {
($width:expr) => {
(($width.wrapping_sub(1) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_831_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 1i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_831_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 1i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(3))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_831_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_831_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_831_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_831_16>() as u32)
};
}

macro_rules! ULL {
() => {
_ULL!()
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VACTIVE__2 {
($vdisplay:expr) => {
((($vdisplay) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_993_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_993_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_16>() as u32)
};
}

macro_rules! VBLANK_END__2 {
($vblank_end:expr) => {
((($vblank_end) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_32>() as u32)
};
}

macro_rules! VBLANK_START__2 {
($vblank_start:expr) => {
((($vblank_start) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1038_16>() as u32)
};
}

macro_rules! VSYNC_END__3 {
($vsync_end:expr) => {
((($vsync_end) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_32>() as u32)
};
}

macro_rules! VSYNC_START {
($vsync_start:expr) => {
((($vsync_start) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_1008_16>() as u32)
};
}

macro_rules! VTOTAL {
($vtotal:expr) => {
((($vtotal) as u32) << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_993_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_993_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(31))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_993_32>() as u32)
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _ICL_PORT_PCS_LN {
($ln:expr) => {
2048i32.wrapping_add($ln.wrapping_mul(256))
};
}

macro_rules! _ICL_PORT_TX_LN {
($ln:expr) => {
2176i32.wrapping_add($ln.wrapping_mul(256))
};
}

/// Given the first two numbers __a and __b of arbitrarily many evenly spaced
/// numbers, pick the 0-based __index'th value.
///
/// Always prefer this over _PICK() if the numbers are evenly spaced.
macro_rules! _PICK_EVEN {
($__index:expr, $__a:expr, $__b:expr) => {
$__a.wrapping_add($__index.wrapping_mul($__b.wrapping_sub($__a)))
};
}

/// Given the first two numbers __a and __b of arbitrarily many evenly spaced
/// numbers, pick the 0-based __index'th value.
///
/// Always prefer this over _PICK() if the numbers are evenly spaced.
macro_rules! _PICK_EVEN__2 {
($__index:expr, $__a:expr, $__b:expr) => {
$__a.wrapping_add((($__index) as i32).wrapping_mul($__b.wrapping_sub($__a)))
};
}

/// Named helper wrappers around _PICK_EVEN() and _PICK_EVEN_2RANGES().
macro_rules! _PIPE__3 {
($pipe:expr, $a:expr, $b:expr) => {
_PICK_EVEN!($pipe, $a, $b)
};
}

macro_rules! _PORT {
($port:expr, $a:expr, $b:expr) => {
_PICK_EVEN__2!($port, $a, $b)
};
}

macro_rules! _TRANS__2 {
($tran:expr, $a:expr, $b:expr) => {
_PICK_EVEN!($tran, $a, $b)
};
}

macro_rules! _ULL {
() => {
_AC!()
};
}

macro_rules! _UL {
() => {
_AC!()
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
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

/// Conversion functions/macros from various pointer types to struct
/// intel_display pointer.
macro_rules! __drm_device_to_intel_display {
($p:expr) => {
{
let __v: *mut intel_display = if !$p.is_null() {
unsafe { __drm_to_display($p) }
} else {
core::ptr::null_mut::<intel_display>()
};
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
macro_rules! __sign_use__7 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u32>() as u64) < 4) as i32)) }
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
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

/// We need to compute the minimum and maximum values representable in a given
/// type. These macros may also be useful elsewhere. It would seem more obvious
/// to do something like:
///
/// #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
/// #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
///
/// Unfortunately, the middle expressions, strictly speaking, have
/// undefined behaviour, and at least some versions of gcc warn about
/// the type_max expression (but not if -fsanitize=undefined is in
/// effect; in that case, the warning is deferred to runtime...).
///
/// The slightly excessive casting in type_min is to make sure the
/// macros also produce sensible values for the exotic type _Bool. [The
/// overflow checkers only almost work for _Bool, but that's
/// a-feature-not-a-bug, since people shouldn't be doing arithmetic on
/// _Bools. Besides, the gcc builtins don't allow _Bool* as third
/// argument.]
///
/// Idea stolen from
/// https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
/// credit to Christian Biere.
macro_rules! __type_half_max {
() => {
1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__7!($ux) & __sign_use!($uy)
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
macro_rules! container_of {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__4 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! fetch_and_zero {
($ptr:expr) => {
{
let mut __v: *mut ref_tracker;
let __v2: *mut ref_tracker = unsafe { *$ptr };
unsafe {
*$ptr = core::ptr::null_mut::<ref_tracker>();
}
__v = __v2;
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Add a ktime_t variable and a scalar nanosecond value.
/// res = kt + nsval:
macro_rules! ktime_add_ns {
($kt:expr, $nsval:expr) => {
(($kt) as u64).wrapping_add($nsval)
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

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dp) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, base.base) as u64
};
}

macro_rules! offsetof__5 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hdmi) as u64
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, base) as u64
};
}

macro_rules! port_name {
($p:expr) => {
(($p) as i32).wrapping_add(65)
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

macro_rules! to_intel_crtc {
($x:expr) => {
{
let mut __v: *mut intel_crtc;
__v = ((unsafe { ((($x) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(intel_crtc) as isize)) }) as *mut ()) as *mut intel_crtc;
__v
}
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__4 {
($p:expr) => {
{
let __v: *mut intel_display = if !$p.is_null() {
unsafe { __drm_to_display($p) }
} else {
core::ptr::null_mut::<intel_display>()
};
__v
}
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*$p).base.dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__5 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*(*$p).uapi.crtc).dev })
};
}

macro_rules! type_max {
() => {
__type_max!()
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

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_5();
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn ktime_get() -> i64;
    fn ktime_get_with_offset(arg0: u32) -> i64;
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn drm_dsc_compute_rc_parameters(arg0: *mut drm_dsc_config) -> i32;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_392();
    fn __compiletime_assert_413();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_419();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn drm_connector_init(arg0: *mut drm_device, arg1: *mut drm_connector, arg2: *const drm_connector_funcs, arg3: i32) -> i32;
    fn drm_connector_cleanup(arg0: *mut drm_connector);
    fn drm_connector_set_panel_orientation_with_quirk(arg0: *mut drm_connector, arg1: i32, arg2: i32, arg3: i32) -> i32;
    fn drm_encoder_init(arg0: *mut drm_device, arg1: *mut drm_encoder, arg2: *const drm_encoder_funcs, arg3: i32, arg4: *const u8, ...) -> i32;
    fn drm_encoder_cleanup(arg0: *mut drm_encoder);
    fn drm_atomic_helper_connector_destroy_state(arg0: *mut drm_connector, arg1: *mut drm_connector_state);
    fn mipi_dsi_packet_format_is_long(arg0: u8) -> bool;
    fn mipi_dsi_create_packet(arg0: *mut mipi_dsi_packet, arg1: *const mipi_dsi_msg) -> i32;
    fn mipi_dsi_set_maximum_return_packet_size(arg0: *mut mipi_dsi_device, arg1: u16) -> i32;
    fn mipi_dsi_dcs_nop(arg0: *mut mipi_dsi_device) -> i32;
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn drm_helper_probe_single_connector_modes(arg0: *mut drm_connector, arg1: u32, arg2: u32) -> i32;
    fn intel_digital_connector_atomic_get_property(arg0: *mut drm_connector, arg1: *const drm_connector_state, arg2: *mut drm_property, arg3: *mut u64) -> i32;
    fn intel_digital_connector_atomic_set_property(arg0: *mut drm_connector, arg1: *mut drm_connector_state, arg2: *mut drm_property, arg3: u64) -> i32;
    fn intel_digital_connector_atomic_check(arg0: *mut drm_connector, arg1: *mut drm_atomic_commit) -> i32;
    fn intel_digital_connector_duplicate_state(arg0: *mut drm_connector) -> *mut drm_connector_state;
    fn intel_backlight_setup(arg0: *mut intel_connector, arg1: i32) -> i32;
    fn intel_backlight_enable(arg0: *const intel_crtc_state, arg1: *const drm_connector_state);
    fn intel_backlight_update(arg0: *mut intel_atomic_state, arg1: *mut intel_encoder, arg2: *const intel_crtc_state, arg3: *const drm_connector_state);
    fn intel_backlight_disable(arg0: *const drm_connector_state);
    fn intel_combo_phy_power_up_lanes(arg0: *mut intel_display, arg1: i32, arg2: bool, arg3: i32, arg4: bool);
    fn intel_connector_alloc() -> *mut intel_connector;
    fn intel_connector_destroy(arg0: *mut drm_connector);
    fn intel_connector_register(arg0: *mut drm_connector) -> i32;
    fn intel_connector_unregister(arg0: *mut drm_connector);
    fn intel_connector_attach_encoder(arg0: *mut intel_connector, arg1: *mut intel_encoder);
    fn intel_connector_get_hw_state(arg0: *mut intel_connector) -> bool;
    fn intel_attach_scaling_mode_property(arg0: *mut drm_connector);
    fn intel_crtc_vblank_on(arg0: *const intel_crtc_state);
    fn intel_crtc_vblank_off(arg0: *const intel_crtc_state);
    fn intel_ddi_get_clock(arg0: *mut intel_encoder, arg1: *mut intel_crtc_state, arg2: *mut intel_dpll);
    fn intel_ddi_port_pll_type(arg0: *mut intel_encoder, arg1: *const intel_crtc_state) -> u32;
    fn icl_ddi_combo_get_pll(arg0: *mut intel_encoder) -> *mut intel_dpll;
    fn intel_display_power_get(arg0: *mut intel_display, arg1: u32) -> *mut ref_tracker;
    fn intel_display_power_get_if_enabled(arg0: *mut intel_display, arg1: u32) -> *mut ref_tracker;
    fn intel_display_power_put_unchecked(arg0: *mut intel_display, arg1: u32);
    fn intel_dmc_wl_get(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14);
    fn intel_dmc_wl_put(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14);
    fn to_intel_uncore(arg0: *mut drm_device) -> *mut intel_uncore;
    fn intel_de_wait_for_set_us(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_de_wait_for_clear_us(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_de_wait_for_set_ms(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_de_wait_for_clear_ms(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn drm_atomic_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn drm_format_info(arg0: u32) -> *const drm_format_info;
    fn intel_bios_init_panel_late(arg0: *mut intel_display, arg1: *mut intel_panel, arg2: *const intel_bios_encoder_data, arg3: *const drm_edid);
    fn intel_bios_get_dsc_params(arg0: *mut intel_encoder, arg1: *mut intel_crtc_state, arg2: i32) -> bool;
    fn intel_bios_encoder_port(arg0: *const intel_bios_encoder_data) -> i32;
    fn intel_cpu_transcoder_mode_valid(arg0: *mut intel_display, arg1: *const drm_display_mode) -> i32;
    fn intel_encoder_destroy(arg0: *mut drm_encoder);
    fn bdw_get_pipe_misc_bpp(arg0: *mut intel_crtc) -> i32;
    fn __drm_to_display(arg0: *mut drm_device) -> *mut intel_display;
    fn intel_dsi_tlpx_ns(arg0: *const intel_dsi) -> i32;
    fn intel_dsi_get_panel_orientation(arg0: *mut intel_connector) -> i32;
    fn intel_dsi_get_modes(arg0: *mut drm_connector) -> i32;
    fn intel_dsi_mode_valid(arg0: *mut drm_connector, arg1: *const drm_display_mode) -> i32;
    fn intel_dsi_host_init(arg0: *mut intel_dsi, arg1: *const mipi_dsi_host_ops, arg2: i32) -> *mut intel_dsi_host;
    fn intel_dsi_wait_panel_power_cycle(arg0: *mut intel_dsi);
    fn intel_dsi_shutdown(arg0: *mut intel_encoder);
    fn intel_dsi_vbt_init(arg0: *mut intel_dsi, arg1: u16) -> bool;
    fn intel_dsi_vbt_exec_sequence(arg0: *mut intel_dsi, arg1: u32);
    fn intel_dsi_log_params(arg0: *mut intel_dsi);
    fn intel_panel_init(arg0: *mut intel_connector, arg1: *const drm_edid) -> i32;
    fn intel_panel_detect(arg0: *mut drm_connector, arg1: bool) -> u32;
    fn intel_panel_preferred_fixed_mode(arg0: *mut intel_connector) -> *const drm_display_mode;
    fn intel_panel_compute_config(arg0: *mut intel_atomic_state, arg1: *mut intel_crtc_state, arg2: *mut intel_connector) -> i32;
    fn intel_panel_add_vbt_lfp_fixed_mode(arg0: *mut intel_connector);
    fn intel_panel_prepare(arg0: *const intel_crtc_state, arg1: *const drm_connector_state);
    fn intel_panel_unprepare(arg0: *const drm_connector_state);
    fn intel_pfit_compute_config(arg0: *mut intel_crtc_state, arg1: *const drm_connector_state) -> i32;
    fn intel_dsc_disable(arg0: *const intel_crtc_state);
    fn intel_dsc_compute_params(arg0: *mut intel_crtc_state) -> i32;
    fn intel_dsc_enable_on_crtc(arg0: *mut intel_crtc_state);
    fn intel_dsc_dsi_pps_write(arg0: *mut intel_encoder, arg1: *const intel_crtc_state);
    fn skl_scaler_disable(arg0: *const intel_crtc_state);
    fn __compiletime_assert_708();
    fn __compiletime_assert_720();
    fn __compiletime_assert_723();
    fn __compiletime_assert_726();
    fn __compiletime_assert_729();
    fn __compiletime_assert_732();
    fn __compiletime_assert_735();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
struct __SlateRecord_types_h_202 {
    refcnt: __SlateRecord_types_h_190,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_4: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_5() };
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
struct io_bitmap {
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
// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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
    __slate_bits_0: __slate_bits::__SlateBits124U0,
}


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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits131U0,
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
struct static_key_mod {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[inline(always)]
fn cpu_relax() {
native_pause();
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
    __slate_bits_0: __slate_bits::__SlateBits132U0,
    pkru: u32,
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
struct perf_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
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
struct plist_head {
    node_list: list_head,
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
struct futex_private_hash {
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
struct sem_undo_list {
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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
struct __SlateRecord_uidgid_types_h_13 {
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
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
struct kmap_ctrl {
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
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
    __slate_bits_0: __slate_bits::__SlateBits73U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits73U1,
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
    __slate_bits_2: __slate_bits::__SlateBits73U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


// Per process flags
// I'm a virtual CPU
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// This thread should not be frozen
// I am kswapd
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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

fn ktime_add_us(kt: i64, usec: u64) -> i64 {
ktime_add_ns!(kt, usec.wrapping_mul(1000)) as i64
}

fn ktime_get_boottime() -> i64 {
unsafe { ktime_get_with_offset(1u32) }
}

fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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


// DSC Rate Control Constants
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dsc_rc_range_parameters {
    range_min_qp: u8,
    range_max_qp: u8,
    range_bpg_offset: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dsc_config {
    line_buf_depth: u8,
    bits_per_component: u8,
    convert_rgb: bool,
    slice_count: u8,
    slice_width: u16,
    slice_height: u16,
    simple_422: bool,
    pic_width: u16,
    pic_height: u16,
    rc_tgt_offset_high: u8,
    rc_tgt_offset_low: u8,
    bits_per_pixel: u16,
    rc_edge_factor: u8,
    rc_quant_incr_limit1: u8,
    rc_quant_incr_limit0: u8,
    initial_xmit_delay: u16,
    initial_dec_delay: u16,
    block_pred_enable: bool,
    first_line_bpg_offset: u8,
    initial_offset: u16,
    rc_buf_thresh: [u16; 14],
    rc_range_params: [drm_dsc_rc_range_parameters; 15],
    rc_model_size: u16,
    flatness_min_qp: u8,
    flatness_max_qp: u8,
    initial_scale_value: u8,
    scale_decrement_interval: u16,
    scale_increment_interval: u16,
    nfl_bpg_offset: u16,
    slice_bpg_offset: u16,
    final_offset: u16,
    vbr_enable: bool,
    mux_word_size: u8,
    slice_chunk_size: u16,
    rc_bits: u16,
    dsc_version_minor: u8,
    dsc_version_major: u8,
    native_422: bool,
    native_420: bool,
    second_line_bpg_offset: u8,
    nsl_bpg_offset: u16,
    second_line_offset_adj: u16,
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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
struct kref {
    refcount: refcount_struct,
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
struct gendisk {
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
struct vm_struct {
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blocking_notifier_head {
    rwsem: rw_semaphore,
    head: *mut notifier_block,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits433U0,
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
struct xol_area {
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
struct rt_mutex_base {
    wait_lock: raw_spinlock,
    waiters: rb_root_cached,
    owner: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex {
    rtmutex: rt_mutex_base,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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
struct drm_mode_object {
    id: u32,
    r#type: u32,
    properties: *mut drm_object_properties,
    refcount: kref,
    free_cb: Option<unsafe extern "C" fn(*mut kref)>,
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
    __slate_bits_0: __slate_bits::__SlateBits500U0,
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
    __slate_bits_0: __slate_bits::__SlateBits502U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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


#[repr(C)]
#[derive(Clone, Copy)]
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_master {
}


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
    __slate_bits_0: __slate_bits::__SlateBits527U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits527U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits527U2,
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
    __slate_bits_3: __slate_bits::__SlateBits527U3,
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct poll_table_struct {
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
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
struct __SlateRecord_uuid_h_17 {
    b: [u8; 16],
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
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct folio_queue {
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_392() };
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
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
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
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


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
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
struct tty_audit_buf {
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_413() };
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits672U0,
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
unsafe { __compiletime_assert_419() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("420: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 420b - ., 3; .popsection\n# {0}", const 420 as i32, options(nostack, att_syntax));
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
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
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
    __slate_bits_0: __slate_bits::__SlateBits733U0,
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
struct posix_acl {
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


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
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
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits524U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irqaction {
    handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    __slate_anon_1: __SlateRecord_interrupt_h_125,
    affinity: *const cpumask,
    next: *mut irqaction,
    thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    thread: *mut task_struct,
    secondary: *mut irqaction,
    irq: u32,
    flags: u32,
    thread_flags: u64,
    thread_mask: u64,
    name: *const u8,
    dir: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_notify {
    irq: u32,
    kref: kref,
    work: work_struct,
    notify: Option<unsafe extern "C" fn(*mut irq_affinity_notify, *const cpumask)>,
    release: Option<unsafe extern "C" fn(*mut kref)>,
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
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
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
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
struct drm_clip_rect {
    x1: u16,
    y1: u16,
    x2: u16,
    y2: u16,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scdc {
    supported: bool,
    read_request: bool,
    scrambling: drm_scrambling,
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
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
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
struct drm_writeback_job {
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
    __slate_bits_0: __slate_bits::__SlateBits917U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct platform_device {
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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
    __slate_bits_0: __slate_bits::__SlateBits929U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_minor {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc {
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
struct dma_fence {
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


fn drm_connector_helper_add(connector: *mut drm_connector, funcs: *const drm_connector_helper_funcs) {
unsafe {
(*connector).helper_private = funcs;
}
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


fn fxp_q4_to_int(val_q4: i32) -> i32 {
val_q4 >> 4i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mipi_dsi_msg {
    channel: u8,
    r#type: u8,
    flags: u16,
    tx_len: u64,
    tx_buf: *const (),
    rx_len: u64,
    rx_buf: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_self_refresh_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mipi_dsi_packet {
    size: u64,
    header: [u8; 4],
    payload_length: u64,
    payload: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mipi_dsi_host_ops {
    attach: Option<unsafe extern "C" fn(*mut mipi_dsi_host, *mut mipi_dsi_device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut mipi_dsi_host, *mut mipi_dsi_device) -> i32>,
    transfer: Option<unsafe extern "C" fn(*mut mipi_dsi_host, *const mipi_dsi_msg) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mipi_dsi_host {
    dev: *mut device,
    ops: *const mipi_dsi_host_ops,
    list: list_head,
}


// transmit data in low power
#[repr(C)]
#[derive(Clone, Copy)]
struct mipi_dsi_device {
    host: *mut mipi_dsi_host,
    dev: device,
    attached: bool,
    name: [u8; 20],
    channel: u32,
    lanes: u32,
    format: u32,
    mode_flags: u64,
    hs_rate: u64,
    lp_rate: u64,
    dsc: *mut drm_dsc_config,
}


fn mipi_dsi_pixel_format_to_bpp(fmt: u32) -> i32 {
match fmt as u32 {
4 => {
return 30;
}
0 | 1 => {
return 24;
}
2 => {
return 18;
}
3 => {
return 16;
}
_ => {
}
}
22i32.wrapping_neg()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scanout_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_connector {
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
// Helper for struct drm_device based WARNs
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_i915_reg_defs_h_14 {
    reg: u32,
}


// These macros can be used on either i915_reg_t or i915_mcr_reg_t since they're
// simply operations on the register's offset and don't care about the MCR vs
// non-MCR nature of the register.
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_config {
    cdclk: u32,
    vco: u32,
    r#ref: u32,
    bypass: u32,
    voltage_level: u8,
    joined_mbus: bool,
}


// Platform group aliases
// g45 and gm45
// mobile platforms
// discrete graphics
// Display ver 2
// Display ver 3
// Display ver 4
// Display ver 5
// Display ver 6
// Display ver 7
// Display ver 8
// Display ver 9
// Display ver 11
// Display ver 12
// Display ver 13
// Display ver 14 (based on GMD ID)
// Display ver 20 (based on GMD ID)
// Display ver 14.1 (based on GMD ID)
// Display ver 30 (based on GMD ID)
// Display ver 35 (based on GMD ID)
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_platforms {
    __slate_anon_0: __SlateRecord_intel_display_device_h_117,
}


// Keep in alphabetical order
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_runtime_info {
    ip: intel_display_ip_ver,
    step: i32,
    step_name: [u8; 3],
    rawclk_freq: u32,
    pipe_mask: u8,
    cpu_transcoder_mask: u16,
    port_mask: u16,
    num_sprites: [u8; 4],
    num_scalers: [u8; 4],
    fbc_mask: u8,
    has_hdcp: bool,
    has_dmc: bool,
    has_dsc: bool,
    edp_typec_support: bool,
    has_dbuf_overlap_detection: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_device_info {
    __runtime_defaults: intel_display_runtime_info,
    abox_mask: u8,
    dbuf: __SlateRecord_intel_display_device_h_320,
    __slate_bits_0: __slate_bits::__SlateBits975U0,
    mmio_offset: u32,
    pipe_offsets: [u32; 9],
    trans_offsets: [u32; 9],
    cursor_offsets: [u32; 4],
    color: __SlateRecord_intel_display_device_h_337,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_bios_encoder_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_params {
    dmc_firmware_path: *mut u8,
    vbt_firmware: *mut u8,
    lvds_channel_mode: i32,
    panel_use_ssc: i32,
    vbt_sdvo_panel_type: i32,
    enable_dc: i32,
    enable_dpt: bool,
    enable_dsb: bool,
    enable_flipq: bool,
    enable_sagv: bool,
    disable_power_well: i32,
    enable_ips: bool,
    invert_brightness: i32,
    edp_vswing: i32,
    enable_dpcd_backlight: i32,
    load_detect_test: bool,
    force_reset_modeset_test: bool,
    disable_display: bool,
    verbose_state_checks: bool,
    nuclear_pageflip: bool,
    enable_dp_mst: bool,
    enable_fbc: i32,
    enable_psr: i32,
    enable_dc_balance: bool,
    enable_panel_replay: i32,
    psr_safest_params: bool,
    enable_psr2_sel_fetch: bool,
    enable_dmc_wl: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_power_domain_mask {
    bits: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dc3co_state {
    lock: mutex,
    allowed: bool,
    trigger: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_power_domains {
    initializing: bool,
    display_core_suspended: bool,
    dc3co_to_dc6: bool,
    power_well_count: i32,
    dc_state: u32,
    target_dc_state: u32,
    allowed_dc_mask: u32,
    init_wakeref: *mut ref_tracker,
    disable_wakeref: *mut ref_tracker,
    lock: mutex,
    domain_use_count: [i32; 76],
    async_put_work: delayed_work,
    async_put_wakeref: *mut ref_tracker,
    async_put_domains: [intel_power_domain_mask; 2],
    async_put_next_delay: i32,
    power_wells: *mut i915_power_well,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_power_domain_set {
    mask: intel_power_domain_mask,
}


fn intel_display_power_put(display: *mut intel_display, domain: u32, wakeref: *mut ref_tracker) {
unsafe { intel_display_power_put_unchecked(display, domain) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dmc_wl {
    lock: spinlock,
    enabled: bool,
    taken: bool,
    refcount: refcount_struct,
    dc_state: u32,
    work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i9xx_dpll_hw_state {
    dpll: u32,
    dpll_md: u32,
    fp0: u32,
    fp1: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hsw_dpll_hw_state {
    wrpll: u32,
    spll: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_dpll_hw_state {
    ctrl1: u32,
    cfgcr1: u32,
    cfgcr2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bxt_dpll_hw_state {
    ebb0: u32,
    ebb4: u32,
    pll0: u32,
    pll1: u32,
    pll2: u32,
    pll3: u32,
    pll6: u32,
    pll8: u32,
    pll9: u32,
    pll10: u32,
    pcsdw12: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_dpll_hw_state {
    cfgcr0: u32,
    cfgcr1: u32,
    div0: u32,
    mg_refclkin_ctl: u32,
    mg_clktop2_coreclkctl1: u32,
    mg_clktop2_hsclkctl: u32,
    mg_pll_div0: u32,
    mg_pll_div1: u32,
    mg_pll_lf: u32,
    mg_pll_frac_lock: u32,
    mg_pll_ssc: u32,
    mg_pll_bias: u32,
    mg_pll_tdc_coldst_bias: u32,
    mg_pll_bias_mask: u32,
    mg_pll_tdc_coldst_bias_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_parent_interface {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_mpllb_state {
    clock: u32,
    ref_control: u32,
    mpllb_cp: u32,
    mpllb_div: u32,
    mpllb_div2: u32,
    mpllb_fracn1: u32,
    mpllb_fracn2: u32,
    mpllb_sscen: u32,
    mpllb_sscstep: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_device_h_117 {
    __slate_anon_0: __SlateRecord_intel_display_device_h_118,
    bitmap: [u64; 2],
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_118 {
    __slate_bits_0: __slate_bits::__SlateBits972U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c10pll_state {
    tx: u8,
    cmn: u8,
    pll: [u8; 20],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_ip_ver {
    ver: u16,
    rel: u16,
    step: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c20pll_state {
    tx: [u16; 3],
    cmn: [u16; 4],
    __slate_anon_2: __SlateRecord_intel_dpll_mgr_h_252,
    vdr: intel_c20pll_vdr_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_320 {
    size: u16,
    slice_mask: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_337 {
    degamma_lut_size: u32,
    gamma_lut_size: u32,
    degamma_lut_tests: u32,
    gamma_lut_tests: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cx0pll_state {
    __slate_anon_0: __SlateRecord_intel_dpll_mgr_h_264,
    lane_count: i32,
    ssc_enabled: bool,
    use_c10: bool,
    tbt_mode: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_power_well {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ref_tracker {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lt_phy_pll_state {
    addr_msb: [u8; 13],
    addr_lsb: [u8; 13],
    data: [[u8; 4]; 13],
    config: [u8; 3],
    ssc_enabled: bool,
    tbt_mode: bool,
    lane_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_hw_state {
    __slate_anon_0: __SlateRecord_intel_dpll_mgr_h_285,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_state {
    pipe_mask: u8,
    hw_state: intel_dpll_hw_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dpll_info {
    name: *const u8,
    funcs: *const intel_dpll_funcs,
    id: i32,
    power_domain: u32,
    always_on: bool,
    is_alt_port_dpll: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll {
    state: intel_dpll_state,
    index: u8,
    active_mask: u8,
    on: bool,
    info: *const dpll_info,
    wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_state_funcs {
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut intel_global_obj) -> *mut intel_global_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut intel_global_obj, *mut intel_global_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_obj {
    head: list_head,
    state: *mut intel_global_state,
    funcs: *const intel_global_state_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_state {
    obj: *mut intel_global_obj,
    state: *mut intel_atomic_state,
    commit: *mut intel_global_commit,
    r#ref: kref,
    changed: bool,
    serialized: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_device_id {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    class: u32,
    class_mask: u32,
    driver_data: u64,
    override_only: u32,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_slot {
    bus: *mut pci_bus,
    list: list_head,
    hotplug: *mut hotplug_slot,
    number: u16,
    __slate_bits_0: __slate_bits::__SlateBits1016U0,
    kobj: kobject,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev {
    bus_list: list_head,
    bus: *mut pci_bus,
    subordinate: *mut pci_bus,
    sysdata: *mut (),
    procent: *mut proc_dir_entry,
    slot: *mut pci_slot,
    devfn: u32,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    class: u32,
    revision: u8,
    hdr_type: u8,
    rcec_ea: *mut rcec_ea,
    rcec: *mut pci_dev,
    devcap: u32,
    rebar_cap: u16,
    pcie_cap: u8,
    msi_cap: u8,
    msix_cap: u8,
    __slate_bits_0: __slate_bits::__SlateBits330U0,
    rom_base_reg: u8,
    pin: u8,
    pcie_flags_reg: u16,
    dma_alias_mask: *mut u64,
    driver: *mut pci_driver,
    dma_mask: u64,
    msi_addr_mask: u64,
    dma_parms: device_dma_parameters,
    current_state: i32,
    pm_cap: u8,
    __slate_bits_1: __slate_bits::__SlateBits330U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits330U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits330U3,
    dev_flags: u16,
    enable_cnt: __SlateRecord_types_h_190,
    pcie_cap_lock: spinlock,
    saved_config_space: [u32; 16],
    saved_cap_space: hlist_head,
    msix_base: *mut (),
    msi_lock: raw_spinlock,
    vpd: pci_vpd,
    link_bwctrl: *mut pcie_bwctrl_data,
    __slate_anon_116: __SlateRecord_pci_h_547,
    ats_cap: u16,
    ats_stu: u8,
    pri_cap: u16,
    pri_reqs_alloc: u32,
    __slate_bits_4: __slate_bits::__SlateBits330U4,
    pasid_cap: u16,
    pasid_features: u16,
    acs_cap: u16,
    acs_capabilities: u16,
    supported_speeds: u8,
    rom: u64,
    romlen: u64,
    priv_flags: u64,
    reset_methods: [u8; 8],
    wake: *mut gpio_desc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus {
    node: list_head,
    parent: *mut pci_bus,
    children: list_head,
    devices: list_head,
    self_: *mut pci_dev,
    slots: list_head,
    resource: [*mut resource; 4],
    resources: list_head,
    busn_res: resource,
    ops: *mut pci_ops,
    sysdata: *mut (),
    procdir: *mut proc_dir_entry,
    number: u8,
    primary: u8,
    max_bus_speed: u8,
    cur_bus_speed: u8,
    name: [u8; 48],
    bridge_ctl: u16,
    bus_flags: u16,
    bridge: *mut device,
    dev: device,
    __slate_bits_0: __slate_bits::__SlateBits463U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_252 {
    mplla: [u16; 10],
    mpllb: [u16; 11],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c20pll_vdr_state {
    custom_width: u8,
    serdes_rate: u8,
    hdmi_rate: u8,
}


// Error values that may be returned by PCI functions
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_ops {
    add_bus: Option<unsafe extern "C" fn(*mut pci_bus) -> i32>,
    remove_bus: Option<unsafe extern "C" fn(*mut pci_bus)>,
    map_bus: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> *mut ()>,
    read: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, *mut u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_264 {
    c10: intel_c10pll_state,
    c20: intel_c20pll_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_285 {
    i9xx: i9xx_dpll_hw_state,
    hsw: hsw_dpll_hw_state,
    skl: skl_dpll_hw_state,
    bxt: bxt_dpll_hw_state,
    icl: icl_dpll_hw_state,
    mpllb: intel_mpllb_state,
    cx0pll: intel_cx0pll_state,
    ltpll: intel_lt_phy_pll_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_error_handlers {
    error_detected: Option<unsafe extern "C" fn(*mut pci_dev, u32) -> u32>,
    mmio_enabled: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    slot_reset: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    reset_prepare: Option<unsafe extern "C" fn(*mut pci_dev)>,
    reset_done: Option<unsafe extern "C" fn(*mut pci_dev)>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev)>,
    cor_error_detected: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_driver {
    name: *const u8,
    id_table: *const pci_device_id,
    probe: Option<unsafe extern "C" fn(*mut pci_dev, *const pci_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut pci_dev)>,
    suspend: Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut pci_dev)>,
    sriov_configure: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_set_msix_vec_count: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_get_vf_total_msix: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    err_handler: *const pci_error_handlers,
    groups: *mut *const attribute_group,
    dev_groups: *mut *const attribute_group,
    driver: device_driver,
    dynids: pci_dynids,
    driver_managed_dma: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsb {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fbc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_commit {
}


// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
fn pci_get_drvdata(pdev: &mut pci_dev) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*(pdev as *mut pci_dev)).dev) }) as *const device)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ilk_wm_values {
    wm_pipe: [u32; 3],
    wm_lp: [u32; 3],
    wm_lp_spr: [u32; 3],
    enable_fbc_wm: bool,
    partitioning: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_pipe_wm {
    plane: [u16; 8],
    fbc: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_sr_wm {
    plane: u16,
    cursor: u16,
    fbc: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_ddl_values {
    plane: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_values {
    pipe: [g4x_pipe_wm; 3],
    sr: g4x_sr_wm,
    ddl: [vlv_wm_ddl_values; 3],
    level: u8,
    cxsr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_wm_values {
    pipe: [g4x_pipe_wm; 2],
    sr: g4x_sr_wm,
    hpll: g4x_sr_wm,
    cxsr: bool,
    hpll_en: bool,
    fbc_en: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_ddb_entry {
    start: u16,
    end: u16,
}


// Amount of PSF GV points, BSpec precisely defines this
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_modeset_funcs {
    get_pipe_config: Option<unsafe extern "C" fn(*mut intel_crtc, *mut intel_crtc_state) -> bool>,
    get_initial_plane_config: Option<unsafe extern "C" fn(*mut intel_crtc, *mut intel_initial_plane_config)>,
    fixup_initial_plane_config: Option<unsafe extern "C" fn(*mut intel_crtc, *const intel_initial_plane_config) -> bool>,
    crtc_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    crtc_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    commit_modeset_enables: Option<unsafe extern "C" fn(*mut intel_atomic_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm_funcs {
    update_wm: Option<unsafe extern "C" fn(*mut intel_display)>,
    compute_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc) -> i32>,
    initial_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    atomic_update_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    optimize_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    compute_global_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state) -> i32>,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_display)>,
    sanitize: Option<unsafe extern "C" fn(*mut intel_display)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio_state {
    encoder: *mut intel_encoder,
    eld: [u8; 128],
    needs_cpu_transcoder_id: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio {
    funcs: *const intel_audio_funcs,
    component: *mut i915_audio_component,
    component_registered: bool,
    mutex: mutex,
    power_refcount: i32,
    freq_cntrl: u32,
    state: [intel_audio_state; 9],
    lpe: __SlateRecord_intel_display_core_h_120,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_global {
    funcs: *const intel_dpll_global_funcs,
    lock: mutex,
    num_dpll: i32,
    dplls: [intel_dpll; 9],
    mgr: *const intel_dpll_mgr,
    ref_clks: __SlateRecord_intel_display_core_h_141,
    pch_ssc_use: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_frontbuffer_tracking {
    lock: spinlock,
    busy_bits: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hotplug {
    funcs: *const intel_hotplug_irq_funcs,
    hotplug_work: delayed_work,
    hpd: *const u32,
    pch_hpd: *const u32,
    stats: [__SlateRecord_intel_display_core_h_170; 15],
    event_bits: u32,
    retry_bits: u32,
    reenable_work: delayed_work,
    long_hpd_pin_mask: u32,
    short_hpd_pin_mask: u32,
    dig_port_work: work_struct,
    poll_init_work: work_struct,
    poll_enabled: bool,
    detection_work_enabled: bool,
    hpd_storm_threshold: u32,
    hpd_short_storm_enabled: u8,
    oob_hotplug_last_state: u64,
    dp_wq: *mut workqueue_struct,
    ignore_long_hpd: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_vbt_data {
    version: u16,
    __slate_bits_0: __slate_bits::__SlateBits1071U0,
    lvds_ssc_freq: i32,
    orientation: i32,
    override_afc_startup: bool,
    override_afc_startup_val: u8,
    crt_ddc_pin: i32,
    display_devices: list_head,
    bdb_blocks: list_head,
    sdvo_mappings: [sdvo_device_mapping; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm {
    funcs: *const intel_wm_funcs,
    pri_latency: [u16; 5],
    spr_latency: [u16; 5],
    cur_latency: [u16; 5],
    skl_latency: [u16; 8],
    __slate_anon_5: __SlateRecord_intel_display_core_h_280,
    num_levels: u8,
    wm_mutex: mutex,
    ipc_enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_audio_component {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display {
    drm: *mut drm_device,
    platform: intel_display_platforms,
    pch_type: i32,
    parent: *const intel_display_parent_interface,
    pipe_list: list_head,
    access: __SlateRecord_intel_display_core_h_314,
    backlight: __SlateRecord_intel_display_core_h_319,
    bw: __SlateRecord_intel_display_core_h_324,
    cdclk: __SlateRecord_intel_display_core_h_340,
    color: __SlateRecord_intel_display_core_h_357,
    dbuf: __SlateRecord_intel_display_core_h_364,
    dbuf_bw: __SlateRecord_intel_display_core_h_371,
    dkl: __SlateRecord_intel_display_core_h_375,
    dmc: __SlateRecord_intel_display_core_h_383,
    dsi: __SlateRecord_intel_display_core_h_388,
    dram: __SlateRecord_intel_display_core_h_393,
    fbc: __SlateRecord_intel_display_core_h_397,
    fbdev: __SlateRecord_intel_display_core_h_408,
    fdi: __SlateRecord_intel_display_core_h_413,
    global: __SlateRecord_intel_display_core_h_421,
    gmbus: __SlateRecord_intel_display_core_h_425,
    hdcp: __SlateRecord_intel_display_core_h_443,
    hti: __SlateRecord_intel_display_core_h_457,
    info: __SlateRecord_intel_display_core_h_467,
    ips: __SlateRecord_intel_display_core_h_475,
    irq: __SlateRecord_intel_display_core_h_479,
    modeset: __SlateRecord_intel_display_core_h_521,
    pkgc: __SlateRecord_intel_display_core_h_526,
    pmdemand: __SlateRecord_intel_display_core_h_532,
    power: __SlateRecord_intel_display_core_h_541,
    pps: __SlateRecord_intel_display_core_h_553,
    properties: __SlateRecord_intel_display_core_h_560,
    quirks: __SlateRecord_intel_display_core_h_565,
    reset: __SlateRecord_intel_display_core_h_569,
    restore: __SlateRecord_intel_display_core_h_573,
    sagv: __SlateRecord_intel_display_core_h_585,
    sbi: __SlateRecord_intel_display_core_h_596,
    snps: __SlateRecord_intel_display_core_h_601,
    state: __SlateRecord_intel_display_core_h_609,
    vlv_clock: __SlateRecord_intel_display_core_h_619,
    wq: __SlateRecord_intel_display_core_h_624,
    dp_tunnel_mgr: *mut drm_dp_tunnel_mgr,
    audio: intel_audio,
    dpll: intel_dpll_global,
    fb_tracking: intel_frontbuffer_tracking,
    hotplug: intel_hotplug,
    opregion: *mut intel_opregion,
    overlay: *mut intel_overlay,
    params: intel_display_params,
    vbt: intel_vbt_data,
    wl: intel_dmc_wl,
    wm: intel_wm,
    psr_dc5_dc6_wa_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_vals {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_color_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dmc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_global_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_mgr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fbdev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fdi_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hotplug_irq_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_mmio_debug {
    lock: spinlock,
    unclaimed_mmio_check: i32,
    saved_mmio_check: i32,
    suspend_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_opregion {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_overlay {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_fw_get {
    force_wake_get: Option<unsafe extern "C" fn(*mut intel_uncore, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_funcs {
    read_fw_domains: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14) -> u32>,
    write_fw_domains: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14) -> u32>,
    mmio_readb: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u8>,
    mmio_readw: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u16>,
    mmio_readl: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u32>,
    mmio_readq: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u64>,
    mmio_writeb: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u8, bool)>,
    mmio_writew: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u16, bool)>,
    mmio_writel: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u32, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_forcewake_range {
    start: u32,
    end: u32,
    domains: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore {
    regs: *mut (),
    i915: *mut drm_i915_private,
    gt: *mut intel_gt,
    rpm: *mut intel_runtime_pm,
    lock: spinlock,
    gsi_offset: u32,
    flags: u32,
    fw_domains_table: *const intel_forcewake_range,
    fw_domains_table_entries: u32,
    shadowed_reg_table: *const i915_mmio_range,
    shadowed_reg_table_entries: u32,
    pmic_bus_access_nb: notifier_block,
    fw_get_funcs: *const intel_uncore_fw_get,
    funcs: intel_uncore_funcs,
    fifo_count: u32,
    fw_domains: u32,
    fw_domains_active: u32,
    fw_domains_timer: u32,
    fw_domains_saved: u32,
    fw_domain: [*mut intel_uncore_forcewake_domain; 16],
    user_forcewake_count: u32,
    debug: *mut intel_uncore_mmio_debug,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_120 {
    platdev: *mut platform_device,
    irq: i32,
}


// register access functions
fn intel_uncore_read(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
unsafe { unsafe { (*uncore).funcs.mmio_readl }.unwrap()(uncore, reg, true) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_141 {
    nssc: i32,
    ssc: i32,
}


fn intel_uncore_read_notrace(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
unsafe { unsafe { (*uncore).funcs.mmio_readl }.unwrap()(uncore, reg, false) }
}

fn intel_uncore_write(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14, val: u32) {
unsafe { unsafe { (*uncore).funcs.mmio_writel }.unwrap()(uncore, reg, val, true) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_170 {
    last_jiffies: u64,
    count: i32,
    blocked_count: i32,
    state: u32,
}


// These are untraced mmio-accessors that are only valid to be used inside
// critical sections, such as inside IRQ handlers, where forcewake is explicitly
// controlled.
//
// Think twice, and think again, before using these.
//
// As an example, these accessors can possibly be used between:
//
// spin_lock_irq(&uncore->lock);
// intel_uncore_forcewake_get__locked();
//
// and
//
// intel_uncore_forcewake_put__locked();
// spin_unlock_irq(&uncore->lock);
//
//
// Note: some registers may not need forcewake held, so
// intel_uncore_forcewake_{get,put} can be omitted, see
// intel_uncore_forcewake_for_reg().
//
// Certain architectures will die if the same cacheline is concurrently accessed
// by different clients (e.g. on Ivybridge). Access to registers should
// therefore generally be serialised, by either the dev_priv->uncore.lock or
// a more localised lock guarding all access to that bank of registers.
fn intel_uncore_rmw(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14, clear: u32, set: u32) -> u32 {
let old: u32 = intel_uncore_read(uncore, reg);
let val: u32 = old & !clear | set;
intel_uncore_write(uncore, reg, val);
old
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sdvo_device_mapping {
    initialized: u8,
    dvo_port: u8,
    target_addr: u8,
    dvo_wiring: u8,
    i2c_pin: u8,
    ddc_pin: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_core_h_280 {
    hw: ilk_wm_values,
    vlv: vlv_wm_values,
    g4x: g4x_wm_values,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_314 {
    any_task_allowed: bool,
    allowed_task: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_319 {
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_324 {
    obj: intel_global_obj,
    max: [intel_bw_info; 6],
    psf_bw: [u32; 3],
    peakbw: [u32; 8],
    num_qgv_points: u8,
    num_psf_gv_points: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_bw_info {
    deratedbw: [u32; 8],
    num_planes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_340 {
    funcs: *const intel_cdclk_funcs,
    hw: intel_cdclk_config,
    table: *const intel_cdclk_vals,
    obj: intel_global_obj,
    max_cdclk_freq: u32,
    max_dotclk_freq: u32,
    skl_preferred_vco_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_357 {
    funcs: *const intel_color_funcs,
    glk_linear_degamma_lut: *mut drm_property_blob,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_364 {
    enabled_slices: u8,
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_371 {
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_375 {
    phy_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_383 {
    dmc: *mut intel_dmc,
    wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_388 {
    mmio_base: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_393 {
    info: *const dram_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dram_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_397 {
    instances: [*mut intel_fbc; 4],
    sys_cache: sys_cache_cfg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sys_cache_cfg {
    lock: mutex,
    id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_408 {
    fbdev: *mut intel_fbdev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_413 {
    funcs: *const intel_fdi_funcs,
    pll_freq: u32,
    rx_config: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_421 {
    obj_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_425 {
    mmio_base: u32,
    mutex: mutex,
    bus: [*mut intel_gmbus; 15],
    wait_queue: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_gmbus {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_443 {
    arbiter: *mut i915_hdcp_arbiter,
    comp_added: bool,
    gsc_context: *mut intel_hdcp_gsc_context,
    hdcp_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp_gsc_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_457 {
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_467 {
    __device_info: *const intel_display_device_info,
    __runtime_info: intel_display_runtime_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_475 {
    false_color: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_479 {
    funcs: *const intel_display_irq_funcs,
    lock: spinlock,
    vlv_display_irqs_enabled: bool,
    vblank_enabled: u8,
    vblank_enable_count: i32,
    vblank_status_last_notified: bool,
    vblank_notify_work: work_struct,
    vlv_imr_mask: u32,
    ilk_de_imr_mask: u32,
    de_pipe_imr_mask: [u32; 4],
    pipestat_irq_mask: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_irq_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_521 {
    funcs: *const intel_modeset_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_526 {
    linetime: [u16; 4],
    disable: [bool; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_532 {
    waitqueue: wait_queue_head,
    lock: mutex,
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_541 {
    domains: i915_power_domains,
    dc3co: intel_dc3co_state,
    chv_phy_control: u32,
    chv_phy_assert: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_553 {
    mmio_base: u32,
    mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_560 {
    broadcast_rgb: *mut drm_property,
    force_audio: *mut drm_property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_565 {
    mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_569 {
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_573 {
    modeset_state: *mut drm_atomic_commit,
    reset_ctx: drm_modeset_acquire_ctx,
    saveDSPARB: u32,
    saveSWF0: [u32; 16],
    saveSWF1: [u32; 16],
    saveSWF3: [u32; 3],
    saveGCDGMBUS: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_585 {
    status: u32,
    block_time_us: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_596 {
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_601 {
    phy_failed_calibration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_609 {
    chv_dpll_md: [u32; 4],
    bxt_phy_grc: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_619 {
    hpll_freq: u32,
    czclk_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_624 {
    modeset: *mut workqueue_struct,
    flip: *mut workqueue_struct,
    cleanup: *mut workqueue_struct,
    unordered: *mut workqueue_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel_mgr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_i915_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_runtime_pm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// When a tracepoint is used, it's name is added to the __tracepoint_check
// section. This section is only used at build time to make sure all
// defined tracepoints are used. It is discarded after the build.
// Make sure the alignment of the structure in the __tracepoints section will
// not add unwanted padding between the beginning of the section and the
// structure. Force alignment to the same alignment as the section start.
//
// When lockdep is enabled, we make sure to always test if RCU is
// "watching" regardless if the tracepoint is enabled or not. Tracepoints
// require RCU to be active, and it should always warn at the tracepoint
// site if it is not watching, as it will need to be active when the
// tracepoint is enabled.
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_gt {
}


fn __to_uncore(display: *mut intel_display) -> *mut intel_uncore {
unsafe { to_intel_uncore((*display).drm) }
}

fn intel_de_read(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
let mut val: u32 = 0;
unsafe { intel_dmc_wl_get(display, reg) };
val = intel_uncore_read(__to_uncore(display), reg);
unsafe { intel_dmc_wl_put(display, reg) };
val
}

fn intel_de_posting_read(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14) {
unsafe { intel_dmc_wl_get(display, reg) };
intel_uncore_read_notrace(__to_uncore(display), reg);
unsafe { intel_dmc_wl_put(display, reg) };
}

fn intel_de_write(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14, val: u32) {
unsafe { intel_dmc_wl_get(display, reg) };
intel_uncore_write(__to_uncore(display), reg, val);
unsafe { intel_dmc_wl_put(display, reg) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct i915_mmio_range {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_forcewake_domain {
    uncore: *mut intel_uncore,
    id: u32,
    mask: u32,
    wake_count: u32,
    active: bool,
    timer: hrtimer,
    reg_set: *mut u32,
    reg_ack: *mut u32,
}


fn intel_de_rmw(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14, clear: u32, set: u32) -> u32 {
let mut val: u32 = 0;
unsafe { intel_dmc_wl_get(display, reg) };
val = intel_uncore_rmw(__to_uncore(display), reg, clear, set);
unsafe { intel_dmc_wl_put(display, reg) };
val
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_constraints {
    list: plist_head,
    target_value: i32,
    default_value: i32,
    no_constraint_value: i32,
    r#type: u32,
    notifiers: *mut blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_request {
    node: plist_node,
    qos: *mut pm_qos_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags_request {
    node: list_head,
    flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags {
    list: list_head,
    effective_flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_constraints {
    min_freq: pm_qos_constraints,
    min_freq_notifiers: blocking_notifier_head,
    max_freq: pm_qos_constraints,
    max_freq_notifiers: blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_qos_request {
    r#type: u32,
    pnode: plist_node,
    qos: *mut freq_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pm_qos_h_109 {
    pnode: plist_node,
    flr: pm_qos_flags_request,
    freq: freq_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos_request {
    r#type: u32,
    data: __SlateRecord_pm_qos_h_109,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
    resume_latency: pm_qos_constraints,
    latency_tolerance: pm_qos_constraints,
    freq: freq_constraints,
    flags: pm_qos_flags,
    resume_latency_req: *mut dev_pm_qos_request,
    latency_tolerance_req: *mut dev_pm_qos_request,
    flags_req: *mut dev_pm_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_common_data {
    state_use_accessors: u32,
    node: u32,
    handler_data: *mut (),
    msi_desc: *mut msi_desc,
    affinity: [cpumask; 1],
    effective_affinity: [cpumask; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
    mask: u32,
    irq: u32,
    hwirq: u64,
    common: *mut irq_common_data,
    chip: *mut irq_chip,
    domain: *mut irq_domain,
    parent_data: *mut irq_data,
    chip_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip {
    name: *const u8,
    irq_startup: Option<unsafe extern "C" fn(*mut irq_data) -> u32>,
    irq_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_enable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_disable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_unmask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_eoi: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_set_affinity: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask, bool) -> i32>,
    irq_pre_redirect: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_retrigger: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_set_type: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_set_wake: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_bus_lock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_bus_sync_unlock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_suspend: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_resume: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_pm_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_calc_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_print_chip: Option<unsafe extern "C" fn(*mut irq_data, *mut seq_file)>,
    irq_request_resources: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_release_resources: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_compose_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_write_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_get_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, *mut bool) -> i32>,
    irq_set_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, bool) -> i32>,
    irq_set_vcpu_affinity: Option<unsafe extern "C" fn(*mut irq_data, *mut ()) -> i32>,
    ipi_send_single: Option<unsafe extern "C" fn(*mut irq_data, u32)>,
    ipi_send_mask: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask)>,
    irq_nmi_setup: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_nmi_teardown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_force_complete_move: Option<unsafe extern "C" fn(*mut irq_data)>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irqstat {
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irq_desc {
    irq_common_data: irq_common_data,
    irq_data: irq_data,
    kstat_irqs: *mut irqstat,
    handle_irq: Option<unsafe extern "C" fn(*mut irq_desc)>,
    action: *mut irqaction,
    status_use_accessors: u32,
    core_internal_state__do_not_mess_with_it: u32,
    depth: u32,
    wake_depth: u32,
    tot_count: u64,
    last_unhandled: u64,
    irq_count: u32,
    irqs_unhandled: u32,
    threads_handled: __SlateRecord_types_h_190,
    threads_handled_last: i32,
    lock: raw_spinlock,
    percpu_enabled: *mut cpumask,
    redirect: irq_redirect,
    affinity_hint: *const cpumask,
    affinity_notify: *mut irq_affinity_notify,
    pending_mask: [cpumask; 1],
    threads_oneshot: u64,
    threads_active: __SlateRecord_types_h_190,
    wait_for_threads: wait_queue_head,
    nr_actions: u32,
    no_suspend_depth: u32,
    cond_suspend_depth: u32,
    force_resume_depth: u32,
    dir: *mut proc_dir_entry,
    refcnt: __SlateRecord_types_h_202,
    rcu: callback_head,
    kobj: kobject,
    request_mutex: mutex,
    parent_irq: i32,
    owner: *mut module,
    name: *const u8,
    resend_node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_regs {
    enable: u64,
    disable: u64,
    mask: u64,
    ack: u64,
    eoi: u64,
    r#type: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_type {
    chip: irq_chip,
    regs: irq_chip_regs,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    r#type: u32,
    mask_cache_priv: u32,
    mask_cache: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_generic {
    lock: raw_spinlock,
    reg_base: *mut (),
    reg_readl: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    reg_writel: Option<unsafe extern "C" fn(u32, *mut ())>,
    suspend: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    resume: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    irq_base: u32,
    irq_cnt: u32,
    mask_cache: u32,
    wake_enabled: u32,
    wake_active: u32,
    num_ct: u32,
    private: *mut (),
    installed: u64,
    unused: u64,
    domain: *mut irq_domain,
    list: list_head,
    chip_types: [irq_chip_type; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
    irqs_per_chip: u32,
    num_chips: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    gc: [*mut irq_chip_generic; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct property {
    name: *mut u8,
    length: i32,
    value: *mut (),
    next: *mut property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
    name: *const u8,
    phandle: u32,
    full_name: *const u8,
    fwnode: fwnode_handle,
    properties: *mut property,
    deadprops: *mut property,
    parent: *mut device_node,
    child: *mut device_node,
    sibling: *mut device_node,
    _flags: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_phandle_args {
    np: *mut device_node,
    args_count: i32,
    args: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec {
    fwnode: *mut fwnode_handle,
    param_count: i32,
    param: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec_info {
    flags: u64,
    affinity: *const cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_ops {
    r#match: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, u32) -> i32>,
    select: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, u32) -> i32>,
    map: Option<unsafe extern "C" fn(*mut irq_domain, u32, u64) -> i32>,
    unmap: Option<unsafe extern "C" fn(*mut irq_domain, u32)>,
    xlate: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, *const u32, u32, *mut u64, *mut u32) -> i32>,
    alloc: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32, *mut ()) -> i32>,
    free: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32)>,
    activate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data, bool) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data)>,
    translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
    get_fwspec_info: Option<unsafe extern "C" fn(*mut irq_fwspec, *mut irq_fwspec_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
    link: list_head,
    name: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    flags: u32,
    mapcount: u32,
    mutex: mutex,
    root: *mut irq_domain,
    fwnode: *mut fwnode_handle,
    bus_token: u32,
    gc: *mut irq_domain_chip_generic,
    dev: *mut device,
    pm_dev: *mut device,
    parent: *mut irq_domain,
    msi_parent_ops: *const msi_parent_ops,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
    hwirq_max: u64,
    revmap_size: u32,
    revmap_tree: xarray,
    revmap: [*mut irq_data; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_chip {
    label: *const u8,
    gpiodev: *mut gpio_device,
    parent: *mut device,
    fwnode: *mut fwnode_handle,
    owner: *mut module,
    request: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    free: Option<unsafe extern "C" fn(*mut gpio_chip, u32)>,
    get_direction: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    direction_input: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    direction_output: Option<unsafe extern "C" fn(*mut gpio_chip, u32, i32) -> i32>,
    get: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    get_multiple: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, *mut u64) -> i32>,
    set: Option<unsafe extern "C" fn(*mut gpio_chip, u32, i32) -> i32>,
    set_multiple: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, *mut u64) -> i32>,
    set_config: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    to_irq: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    dbg_show: Option<unsafe extern "C" fn(*mut seq_file, *mut gpio_chip)>,
    init_valid_mask: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, u32) -> i32>,
    add_pin_ranges: Option<unsafe extern "C" fn(*mut gpio_chip) -> i32>,
    en_hw_timestamp: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    dis_hw_timestamp: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    base: i32,
    ngpio: u16,
    offset: u16,
    names: *const *const u8,
    can_sleep: bool,
}


#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_578: [u8; 19] = [105, 57, 49, 53, 46, 105, 109, 112, 111, 114, 116, 95, 110, 115, 61, 80, 87, 77, 0];

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pwm_h_348 {
    nonatomic_lock: mutex,
    atomic_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_args {
    period: u64,
    polarity: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_user_namespace_h_26 {
    __slate_anon_0: __SlateRecord_user_namespace_h_27,
    __slate_anon_1: __SlateRecord_user_namespace_h_31,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_27 {
    extent: [uid_gid_extent; 5],
    nr_extents: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_31 {
    forward: *mut uid_gid_extent,
    reverse: *mut uid_gid_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct binfmt_misc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_waveform {
    period_length_ns: u64,
    duty_length_ns: u64,
    duty_offset_ns: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_state {
    period: u64,
    duty_cycle: u64,
    polarity: u32,
    enabled: bool,
    usage_power: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_device {
    label: *const u8,
    flags: u64,
    hwpwm: u32,
    chip: *mut pwm_chip,
    args: pwm_args,
    state: pwm_state,
    last: pwm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_capture {
    period: u32,
    duty_cycle: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_ops {
    request: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device) -> i32>,
    free: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device)>,
    capture: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut pwm_capture, u64) -> i32>,
    sizeof_wfhw: u64,
    round_waveform_tohw: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const pwm_waveform, *mut ()) -> i32>,
    round_waveform_fromhw: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const (), *mut pwm_waveform) -> i32>,
    read_waveform: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut ()) -> i32>,
    write_waveform: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const ()) -> i32>,
    apply: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const pwm_state) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut pwm_state) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_chip {
    dev: device,
    cdev: cdev,
    ops: *const pwm_ops,
    owner: *mut module,
    id: u32,
    npwm: u32,
    of_xlate: Option<unsafe extern "C" fn(*mut pwm_chip, *const of_phandle_args) -> *mut pwm_device>,
    atomic: bool,
    gpio: gpio_chip,
    uses_pwmchip_alloc: bool,
    operational: bool,
    __slate_anon_11: __SlateRecord_pwm_h_348,
    pwms: [pwm_device; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
    version: u16,
    ac_exitcode: u32,
    ac_flag: u8,
    ac_nice: u8,
    cpu_count: u64,
    cpu_delay_total: u64,
    blkio_count: u64,
    blkio_delay_total: u64,
    swapin_count: u64,
    swapin_delay_total: u64,
    cpu_run_real_total: u64,
    cpu_run_virtual_total: u64,
    ac_comm: [u8; 32],
    ac_sched: u8,
    ac_pad: [u8; 3],
    __slate_pad_15: [u8; 4],
    ac_uid: u32,
    ac_gid: u32,
    ac_pid: u32,
    ac_ppid: u32,
    ac_btime: u32,
    ac_etime: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
    coremem: u64,
    virtmem: u64,
    hiwater_rss: u64,
    hiwater_vm: u64,
    read_char: u64,
    write_char: u64,
    read_syscalls: u64,
    write_syscalls: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
    nvcsw: u64,
    nivcsw: u64,
    ac_utimescaled: u64,
    ac_stimescaled: u64,
    cpu_scaled_run_real_total: u64,
    freepages_count: u64,
    freepages_delay_total: u64,
    thrashing_count: u64,
    thrashing_delay_total: u64,
    ac_btime64: u64,
    compact_count: u64,
    compact_delay_total: u64,
    ac_tgid: u32,
    ac_tgetime: u64,
    ac_exe_dev: u64,
    ac_exe_inode: u64,
    wpcopy_count: u64,
    wpcopy_delay_total: u64,
    irq_count: u64,
    irq_delay_total: u64,
    cpu_delay_max: u64,
    cpu_delay_min: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    cpu_delay_max_ts: __kernel_timespec,
    blkio_delay_max_ts: __kernel_timespec,
    swapin_delay_max_ts: __kernel_timespec,
    freepages_delay_max_ts: __kernel_timespec,
    thrashing_delay_max_ts: __kernel_timespec,
    compact_delay_max_ts: __kernel_timespec,
    wpcopy_delay_max_ts: __kernel_timespec,
    irq_delay_max_ts: __kernel_timespec,
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
struct cgroup_taskset {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
    uid_map: uid_gid_map,
    gid_map: uid_gid_map,
    projid_map: uid_gid_map,
    parent: *mut user_namespace,
    level: i32,
    owner: __SlateRecord_uidgid_types_h_9,
    group: __SlateRecord_uidgid_types_h_13,
    ns: ns_common,
    flags: u64,
    parent_could_setfcap: bool,
    keyring_name_list: list_head,
    user_keyring_register: *mut key,
    keyring_sem: rw_semaphore,
    work: work_struct,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    ucounts: *mut ucounts,
    ucount_max: [i64; 11],
    rlimit_max: [i64; 4],
    binfmt_misc: *mut binfmt_misc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
    node: hlist_nulls_node,
    ns: *mut user_namespace,
    uid: __SlateRecord_uidgid_types_h_9,
    rcu: callback_head,
    count: __SlateRecord_types_h_202,
    ucount: [__SlateRecord_types_h_197; 11],
    rlimit: [__SlateRecord_types_h_197; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cgroup_defs_h_644 {
    __slate_anon_0: __SlateRecord_cgroup_defs_h_645,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_646,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645 {
    __empty_ancestors: __SlateRecord_cgroup_defs_h_645_2,
    ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_646 {
    _root_ancestor: *mut cgroup,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_648,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648 {
    __empty__low_ancestors: __SlateRecord_cgroup_defs_h_648_2,
    _low_ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
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
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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
struct psi_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_file {
    kn: *mut kernfs_node,
    notified_at: u64,
    notify_timer: timer_list,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_i2c_h_551 {
    xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_i2c_h_557 {
    xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys_state {
    cgroup: *mut cgroup,
    ss: *mut cgroup_subsys,
    refcnt: percpu_ref,
    rstat_cpu: *mut css_rstat_cpu,
    sibling: list_head,
    children: list_head,
    id: i32,
    flags: u32,
    serial_nr: u64,
    online_cnt: __SlateRecord_types_h_190,
    destroy_work: work_struct,
    destroy_rwork: rcu_work,
    parent: *mut cgroup_subsys_state,
    nr_descendants: i32,
    nr_populated_csets: i32,
    nr_populated_children: i32,
    kill_finish_work: work_struct,
    rstat_flush_next: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
    subsys: [*mut cgroup_subsys_state; 14],
    refcount: refcount_struct,
    dom_cset: *mut css_set,
    dfl_cgrp: *mut cgroup,
    nr_tasks: i32,
    tasks: list_head,
    mg_tasks: list_head,
    dying_tasks: list_head,
    task_iters: list_head,
    e_cset_node: [list_head; 14],
    threaded_csets: list_head,
    threaded_csets_node: list_head,
    hlist: hlist_node,
    cgrp_links: list_head,
    mg_src_preload_node: list_head,
    mg_dst_preload_node: list_head,
    mg_node: list_head,
    mg_src_cgrp: *mut cgroup,
    mg_dst_cgrp: *mut cgroup,
    mg_dst_cset: *mut css_set,
    dead: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_base_stat {
    cputime: task_cputime,
    ntime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_rstat_cpu {
    updated_children: *mut cgroup_subsys_state,
    updated_next: *mut cgroup_subsys_state,
    lnode: llist_node,
    owner: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_rstat_base_cpu {
    bsync: u64_stats_sync,
    bstat: cgroup_base_stat,
    last_bstat: cgroup_base_stat,
    subtree_bstat: cgroup_base_stat,
    last_subtree_bstat: cgroup_base_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_freezer_state {
    freeze: bool,
    e_freeze: bool,
    nr_frozen_descendants: i32,
    nr_frozen_tasks: i32,
    freeze_seq: seqcount_spinlock,
    freeze_start_nsec: u64,
    frozen_nsec: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup {
    self_: cgroup_subsys_state,
    flags: u64,
    level: i32,
    max_depth: i32,
    nr_descendants: i32,
    nr_dying_descendants: i32,
    max_descendants: i32,
    nr_populated_domain_children: i32,
    nr_populated_threaded_children: i32,
    nr_threaded_children: i32,
    kill_seq: u32,
    kn: *mut kernfs_node,
    procs_file: cgroup_file,
    events_file: cgroup_file,
    psi_files: [cgroup_file; 0],
    subtree_control: u32,
    subtree_ss_mask: u32,
    old_subtree_control: u32,
    old_subtree_ss_mask: u32,
    subsys: [*mut cgroup_subsys_state; 14],
    nr_dying_subsys: [i32; 14],
    root: *mut cgroup_root,
    cset_links: list_head,
    e_csets: [list_head; 14],
    dom_cgrp: *mut cgroup,
    old_dom_cgrp: *mut cgroup,
    rstat_base_cpu: *mut cgroup_rstat_base_cpu,
    _pad_: cacheline_padding,
    last_bstat: cgroup_base_stat,
    bstat: cgroup_base_stat,
    prev_cputime: prev_cputime,
    pidlists: list_head,
    pidlist_mutex: mutex,
    offline_waitq: wait_queue_head,
    release_agent_work: work_struct,
    psi: *mut psi_group,
    bpf: cgroup_bpf,
    freezer: cgroup_freezer_state,
    __slate_anon_38: __SlateRecord_cgroup_defs_h_644,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cec_adapter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_root {
    kf_root: *mut kernfs_root,
    subsys_mask: u32,
    hierarchy_id: i32,
    root_list: list_head,
    rcu: callback_head,
    nr_cgrps: __SlateRecord_types_h_190,
    flags: u32,
    release_agent_path: [u8; 4096],
    name: [u8; 64],
    cgrp: cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cftype {
    name: [u8; 64],
    private: u64,
    max_write_len: u64,
    flags: u32,
    file_offset: u32,
    ss: *mut cgroup_subsys,
    node: list_head,
    kf_ops: *mut kernfs_ops,
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    read_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> u64>,
    read_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> i64>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    write_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, u64) -> i32>,
    write_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, i64) -> i32>,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    lockdep_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys {
    css_alloc: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> *mut cgroup_subsys_state>,
    css_online: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> i32>,
    css_offline: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_released: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_free: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_reset: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_killed: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_rstat_flush: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, i32)>,
    css_extra_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    css_local_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    can_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset) -> i32>,
    cancel_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    can_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set) -> i32>,
    cancel_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set)>,
    fork: Option<unsafe extern "C" fn(*mut task_struct)>,
    exit: Option<unsafe extern "C" fn(*mut task_struct)>,
    release: Option<unsafe extern "C" fn(*mut task_struct)>,
    bind: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    __slate_bits_0: __slate_bits::__SlateBits1193U0,
    id: i32,
    name: *const u8,
    legacy_name: *const u8,
    root: *mut cgroup_root,
    css_idr: idr,
    cfts: list_head,
    dfl_cftypes: *mut cftype,
    legacy_cftypes: *mut cftype,
    depends_on: u32,
    rstat_ss_lock: spinlock,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
    ns: ns_common,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    root_cset: *mut css_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bdi_writeback {
    bdi: *mut backing_dev_info,
    state: u64,
    last_old_flush: u64,
    b_dirty: list_head,
    b_io: list_head,
    b_more_io: list_head,
    b_dirty_time: list_head,
    list_lock: spinlock,
    writeback_inodes: __SlateRecord_types_h_190,
    stat: [percpu_counter; 5],
    bw_time_stamp: u64,
    dirtied_stamp: u64,
    written_stamp: u64,
    write_bandwidth: u64,
    avg_write_bandwidth: u64,
    dirty_ratelimit: u64,
    balanced_dirty_ratelimit: u64,
    completions: fprop_local_percpu,
    dirty_exceeded: i32,
    start_all_reason: u32,
    work_lock: spinlock,
    work_list: list_head,
    dwork: delayed_work,
    bw_dwork: delayed_work,
    bdi_node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
    id: u64,
    rb_node: rb_node,
    bdi_list: list_head,
    ra_pages: u64,
    io_pages: u64,
    refcnt: kref,
    capabilities: u32,
    min_ratio: u32,
    max_ratio: u32,
    max_prop_frac: u32,
    tot_write_bandwidth: __SlateRecord_types_h_197,
    last_bdp_sleep: u64,
    wb: bdi_writeback,
    wb_list: list_head,
    wb_waitq: wait_queue_head,
    dev: *mut device,
    dev_name: [u8; 64],
    owner: *mut device,
    debug_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
    bd_start_sect: u64,
    bd_nr_sectors: u64,
    bd_disk: *mut gendisk,
    bd_queue: *mut request_queue,
    bd_stats: *mut disk_stats,
    bd_stamp: u64,
    __bd_flags: __SlateRecord_types_h_190,
    bd_dev: u32,
    bd_mapping: *mut address_space,
    bd_openers: __SlateRecord_types_h_190,
    bd_size_lock: spinlock,
    bd_claiming: *mut (),
    bd_holder: *mut (),
    bd_holder_ops: *const blk_holder_ops,
    bd_holder_lock: mutex,
    bd_holders: i32,
    bd_holder_dir: *mut kobject,
    bd_fsfreeze_count: __SlateRecord_types_h_190,
    bd_fsfreeze_mutex: mutex,
    bd_meta_info: *mut partition_meta_info,
    bd_writers: i32,
    bd_security: *mut (),
    bd_device: device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits641U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_sequential_cluster {
    next: [u32; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
    users: percpu_ref,
    flags: u64,
    prio: i16,
    list: plist_node,
    r#type: i8,
    max: u32,
    cluster_info: *mut swap_cluster_info,
    free_clusters: list_head,
    full_clusters: list_head,
    nonfull_clusters: [list_head; 1],
    frag_clusters: [list_head; 1],
    pages: u32,
    inuse_pages: __SlateRecord_types_h_197,
    global_cluster: *mut swap_sequential_cluster,
    global_cluster_lock: spinlock,
    swap_extent_root: rb_root,
    bdev: *mut block_device,
    swap_file: *mut file,
    comp: completion,
    lock: spinlock,
    discard_work: work_struct,
    reclaim_work: work_struct,
    discard_clusters: list_head,
    avail_list: plist_node,
    ops: *const swap_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_msg {
    addr: u16,
    flags: u16,
    len: u16,
    buf: *mut u8,
}


// Data for SMBus Messages
// As specified in SMBus standard
#[repr(C)]
#[derive(Clone, Copy)]
union i2c_smbus_data {
    byte: u8,
    word: u16,
    block: [u8; 34],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_algorithm {
    __slate_anon_0: __SlateRecord_i2c_h_551,
    __slate_anon_1: __SlateRecord_i2c_h_557,
    smbus_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    smbus_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    functionality: Option<unsafe extern "C" fn(*mut i2c_adapter) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_lock_operations {
    lock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
    trylock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32) -> i32>,
    unlock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_fourcc_h_77 {
    cpp: [u8; 4],
    char_per_block: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_gem_object {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_bus_recovery_info {
    recover_bus: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    get_scl: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_scl: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_sda: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_sda: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_bus_free: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    prepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    unprepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    scl_gpiod: *mut gpio_desc,
    sda_gpiod: *mut gpio_desc,
    pinctrl: *mut pinctrl,
    pins_default: *mut pinctrl_state,
    pins_gpio: *mut pinctrl_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backlight_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct display_timing {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter_quirks {
    flags: u64,
    max_num_msgs: i32,
    max_write_len: u16,
    max_read_len: u16,
    max_comb_1st_msg_len: u16,
    max_comb_2nd_msg_len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter {
    owner: *mut module,
    class: u32,
    algo: *const i2c_algorithm,
    algo_data: *mut (),
    lock_ops: *const i2c_lock_operations,
    bus_lock: rt_mutex,
    mux_lock: rt_mutex,
    timeout: i32,
    retries: i32,
    dev: device,
    locked_flags: u64,
    nr: i32,
    name: [u8; 48],
    dev_released: completion,
    userspace_clients_lock: mutex,
    userspace_clients: list_head,
    bus_recovery_info: *mut i2c_bus_recovery_info,
    quirks: *const i2c_adapter_quirks,
    host_notify_domain: *mut irq_domain,
    bus_regulator: *mut regulator,
    debugfs: *mut dentry,
    addrs_in_instantiation: [u64; 2],
}


// Adapter locking functions, exported for shared pin cases
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_vsc_sdp {
    sdp_type: u8,
    revision: u8,
    length: u8,
    pixelformat: u32,
    colorimetry: u32,
    bpc: i32,
    dynamic_range: u32,
    content_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_as_sdp {
    sdp_type: u8,
    revision: u8,
    length: u8,
    vtotal: i32,
    target_rr: i32,
    duration_incr_ms: i32,
    duration_decr_ms: i32,
    target_rr_divider: bool,
    mode: u32,
    coasting_vtotal: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux_msg {
    address: u32,
    request: u8,
    reply: u8,
    buffer: *mut (),
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux_cec {
    lock: mutex,
    adap: *mut cec_adapter,
    connector: *mut drm_connector,
    unregister_work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux {
    name: *const u8,
    ddc: i2c_adapter,
    dev: *mut device,
    drm_dev: *mut drm_device,
    crtc: *mut drm_crtc,
    hw_mutex: mutex,
    crc_work: work_struct,
    crc_count: u8,
    transfer: Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_dp_aux_msg) -> i64>,
    wait_hpd_asserted: Option<unsafe extern "C" fn(*mut drm_dp_aux, u64) -> i32>,
    i2c_nack_count: u32,
    i2c_defer_count: u32,
    cec: drm_dp_aux_cec,
    is_remote: bool,
    powered_down: bool,
    no_zero_sized: bool,
    dpcd_probe_disabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_dpcd_ident {
    oui: [u8; 3],
    device_id: [u8; 6],
    hw_rev: u8,
    sw_major_rev: u8,
    sw_minor_rev: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_desc {
    ident: drm_dp_dpcd_ident,
    quirks: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edp_backlight_info {
    pwmgen_bit_count: u8,
    pwm_freq_pre_divider: u8,
    max: u32,
    __slate_bits_0: __slate_bits::__SlateBits1237U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_phy_test_params {
    link_rate: i32,
    num_lanes: u8,
    phy_pattern: u8,
    hbr2_reset: [u8; 2],
    custom80: [u8; 10],
    enhanced_frame_cap: bool,
}


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
    __slate_bits_0: __slate_bits::__SlateBits881U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_50 {
    __slate_bits_0: __slate_bits::__SlateBits1297U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_84 {
    __slate_bits_0: __slate_bits::__SlateBits1298U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_102 {
    __slate_bits_0: __slate_bits::__SlateBits1299U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits1300U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_122 {
    __slate_bits_0: __slate_bits::__SlateBits1301U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_140 {
    __slate_bits_0: __slate_bits::__SlateBits1302U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_port {
    topology_kref: kref,
    malloc_kref: kref,
    port_num: u8,
    input: bool,
    mcs: bool,
    ddps: bool,
    pdt: u8,
    ldps: bool,
    dpcd_rev: u8,
    num_sdp_streams: u8,
    num_sdp_stream_sinks: u8,
    full_pbn: u16,
    next: list_head,
    mstb: *mut drm_dp_mst_branch,
    aux: drm_dp_aux,
    passthrough_aux: *mut drm_dp_aux,
    parent: *mut drm_dp_mst_branch,
    connector: *mut drm_connector,
    mgr: *mut drm_dp_mst_topology_mgr,
    cached_edid: *const drm_edid,
    fec_capable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cec_notifier {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_ddi_buf_trans {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_caps {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_training {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_objs_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp_shim {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_tc_port {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_hdr {
    lct: u8,
    lcr: u8,
    rad: [u8; 8],
    broadcast: bool,
    path_msg: bool,
    msg_len: u8,
    somt: bool,
    eomt: bool,
    seqno: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_rx {
    chunk: [u8; 48],
    msg: [u8; 256],
    curchunk_len: u8,
    curchunk_idx: u8,
    curchunk_hdrlen: u8,
    curlen: u8,
    have_somt: bool,
    have_eomt: bool,
    initial_hdr: drm_dp_sideband_msg_hdr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_branch {
    topology_kref: kref,
    malloc_kref: kref,
    destroy_next: list_head,
    rad: [u8; 8],
    lct: u8,
    num_ports: i32,
    ports: list_head,
    port_parent: *mut drm_dp_mst_port,
    mgr: *mut drm_dp_mst_topology_mgr,
    link_address_sent: bool,
    guid: __SlateRecord_uuid_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_345 {
    max_link_rate: i32,
    rate: i32,
    lanes: i32,
    preemphasis: i32,
    vswing: i32,
    bpp: i32,
    pps: intel_pps_delays,
    drrs_msa_timing_delay: u8,
    low_vswing: bool,
    hobl: bool,
    dsc_disable: bool,
    pipe_joiner_enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_360 {
    enable: bool,
    full_link: bool,
    require_aux_wakeup: bool,
    idle_frames: i32,
    tp1_wakeup_time_us: i32,
    tp2_tp3_wakeup_time_us: i32,
    psr2_tp2_tp3_wakeup_time_us: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_370 {
    pwm_freq_hz: u16,
    brightness_precision_bits: u16,
    hdr_dpcd_refresh_timeout: u16,
    present: bool,
    active_low_pwm: bool,
    min_brightness: u8,
    controller: i8,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_382 {
    panel_id: u16,
    config: *mut mipi_config,
    pps: *mut mipi_pps_data,
    bl_ports: u16,
    cabc_ports: u16,
    seq_version: u8,
    size: u32,
    data: *mut u8,
    sequence: [*const u8; 12],
    deassert_seq: *mut u8,
    orientation: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_407 {
    present: bool,
    level: u32,
    min: u32,
    max: u32,
    enabled: bool,
    combination_mode: bool,
    active_low_pwm: bool,
    alternate_pwm_increment: bool,
    pwm_level_min: u32,
    pwm_level_max: u32,
    pwm_enabled: bool,
    util_pin_active_low: bool,
    controller: u8,
    pwm: *mut pwm_device,
    pwm_state: pwm_state,
    edp: __SlateRecord_intel_display_types_h_427,
    device: *mut backlight_device,
    funcs: *const intel_panel_bl_funcs,
    pwm_funcs: *const intel_panel_bl_funcs,
    power: Option<unsafe extern "C" fn(*mut intel_connector, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_427 {
    vesa: __SlateRecord_intel_display_types_h_428,
    intel_cap: __SlateRecord_intel_display_types_h_432,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_428 {
    info: drm_edp_backlight_info,
    luminance_control_support: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_432 {
    sdr_uses_aux: bool,
    supports_2084_decode: bool,
    supports_2020_gamut: bool,
    supports_segmented_backlight: bool,
    supports_sdp_colorimetry: bool,
    supports_tone_mapping: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_cbs {
    add_connector: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, *const u8) -> *mut drm_connector>,
    poll_hpd_irq: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_559 {
    dsc_decompression_aux: *mut drm_dp_aux,
    dsc_dpcd: [u8; 16],
    fec_capability: u8,
    __slate_bits_0: __slate_bits::__SlateBits1329U0,
    dsc_branch_caps: __SlateRecord_intel_display_types_h_568,
    panel_replay_caps: __SlateRecord_intel_display_types_h_576,
    psr_caps: __SlateRecord_intel_display_types_h_588,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_568 {
    overall_throughput: __SlateRecord_intel_display_types_h_569,
    max_line_width: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_569 {
    rgb_yuv444: i32,
    yuv422_420: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_576 {
    dpcd: [u8; 7],
    support: bool,
    su_support: bool,
    dsc_support: u32,
    su_w_granularity: u16,
    su_y_granularity: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_588 {
    dpcd: [u8; 2],
    intel_wa_dpcd: u8,
    support: bool,
    su_support: bool,
    su_w_granularity: u16,
    su_y_granularity: u16,
    sync_latency: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_602 {
    port: *mut drm_dp_mst_port,
    dp: *mut intel_dp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_607 {
    force_bpp_x16: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_mgr {
    base: drm_private_obj,
    dev: *mut drm_device,
    cbs: *const drm_dp_mst_topology_cbs,
    max_dpcd_transaction_bytes: i32,
    aux: *mut drm_dp_aux,
    max_payloads: i32,
    conn_base_id: i32,
    up_req_recv: drm_dp_sideband_msg_rx,
    down_rep_recv: drm_dp_sideband_msg_rx,
    lock: mutex,
    probe_lock: mutex,
    __slate_bits_0: __slate_bits::__SlateBits1248U0,
    payload_count: u8,
    next_start_slot: u8,
    mst_primary: *mut drm_dp_mst_branch,
    dpcd: [u8; 15],
    sink_count: u8,
    funcs: *const drm_private_state_funcs,
    qlock: mutex,
    tx_msg_downq: list_head,
    tx_waitq: wait_queue_head,
    work: work_struct,
    tx_work: work_struct,
    destroy_port_list: list_head,
    destroy_branch_device_list: list_head,
    delayed_destroy_lock: mutex,
    delayed_destroy_wq: *mut workqueue_struct,
    delayed_destroy_work: work_struct,
    up_req_list: list_head,
    up_req_lock: mutex,
    up_req_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_tunnel_inherited_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel_ref {
    tunnel: *mut drm_dp_tunnel,
    tracker: *mut ref_tracker,
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
struct __SlateRecord_intel_display_types_h_853 {
    wm0: skl_wm_level,
    trans_wm: skl_wm_level,
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
struct drm_panel_funcs {
    prepare: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    enable: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    disable: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    unprepare: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    get_modes: Option<unsafe extern "C" fn(*mut drm_panel, *mut drm_connector) -> i32>,
    get_orientation: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    get_timings: Option<unsafe extern "C" fn(*mut drm_panel, u32, *mut display_timing) -> i32>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_panel, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_panel {
    dev: *mut device,
    backlight: *mut backlight_device,
    funcs: *const drm_panel_funcs,
    connector_type: i32,
    list: list_head,
    followers: list_head,
    follower_lock: mutex,
    prepare_prev_first: bool,
    prepared: bool,
    enabled: bool,
    container: *mut (),
    refcount: kref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_work {
    base: kthread_work,
    vblank: *mut drm_vblank_crtc,
    count: u64,
    cancelling: i32,
    node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_cert_rx {
    receiver_id: [u8; 5],
    kpub_rx: [u8; 131],
    reserved: [u8; 2],
    dcp_signature: [u8; 384],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_899 {
    ilk: __SlateRecord_intel_display_types_h_922,
    skl: __SlateRecord_intel_display_types_h_927,
    vlv: __SlateRecord_intel_display_types_h_948,
    g4x: __SlateRecord_intel_display_types_h_955,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_922 {
    intermediate: intel_pipe_wm,
    optimal: intel_pipe_wm,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_927 {
    raw: skl_pipe_wm,
    optimal: skl_pipe_wm,
    ddb: skl_ddb_entry,
    plane_ddb: [skl_ddb_entry; 8],
    plane_ddb_y: [skl_ddb_entry; 8],
    plane_min_ddb: [u16; 8],
    plane_interim_ddb: [u16; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_948 {
    raw: [g4x_pipe_wm; 3],
    intermediate: vlv_wm_state,
    optimal: vlv_wm_state,
    fifo_state: vlv_fifo_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_955 {
    raw: [g4x_pipe_wm; 3],
    intermediate: g4x_wm_state,
    optimal: g4x_wm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_streamid_type {
    stream_id: u8,
    stream_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_tx_caps {
    version: u8,
    tx_cap_mask: [u8; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_init {
    msg_id: u8,
    r_tx: [u8; 8],
    tx_caps: hdcp2_tx_caps,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1036 {
    active: bool,
    enable: bool,
    degamma_lut: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    mode: drm_display_mode,
    pipe_mode: drm_display_mode,
    adjusted_mode: drm_display_mode,
    background_color: u32,
    scaling_filter: u32,
    sharpness_strength: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_port_dpll {
    pll: *mut intel_dpll,
    hw_state: intel_dpll_hw_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1171 {
    ctrl: u32,
    div: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1230 {
    control: u32,
    pgm_ratios: u32,
    lvds_border_bits: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1237 {
    casf: intel_casf,
    dst: drm_rect,
    enabled: bool,
    force_thru: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1277 {
    csc_mode: u32,
    cgm_mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1303 {
    enable: u32,
    gcp: u32,
    avi: hdmi_infoframe,
    spd: hdmi_infoframe,
    hdmi: hdmi_infoframe,
    drm: hdmi_infoframe,
    vsc: drm_dp_vsc_sdp,
    as_sdp: drm_dp_as_sdp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1347 {
    compression_enabled_on_link: bool,
    compression_enable: bool,
    slice_config: intel_dsc_slice_config,
    compressed_bpp_x16: u16,
    config: drm_dsc_config,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsc_slice_config {
    pipes_per_line: i32,
    streams_per_pipe: i32,
    slices_per_stream: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1401 {
    enable: bool,
    in_range: bool,
    pipeline_full: u8,
    flipline: u16,
    vmin: u16,
    vmax: u16,
    guardband: u16,
    vsync_end: u32,
    vsync_start: u32,
    dc_balance: __SlateRecord_intel_display_types_h_1406,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1406 {
    enable: bool,
    vmin: u16,
    vmax: u16,
    guardband: u16,
    slope: u16,
    max_increase: u16,
    max_decrease: u16,
    vblank_target: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1416 {
    enable: bool,
    cmrr_n: u64,
    cmrr_m: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1422 {
    enable: bool,
    link_count: u8,
    pixel_overlap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1437 {
    io_wake_lines: u8,
    fast_wake_lines: u8,
    check_entry_lines: u8,
    aux_less_wake_lines: u8,
    silence_period_sym_clocks: u8,
    lfps_half_cycle_num_of_syms: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_cert {
    msg_id: u8,
    cert_rx: hdcp2_cert_rx,
    r_rx: [u8; 8],
    rx_caps: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_no_stored_km {
    msg_id: u8,
    e_kpub_km: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1535 {
    active: __SlateRecord_intel_display_types_h_1537,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1537 {
    ilk: intel_pipe_wm,
    vlv: vlv_wm_state,
    g4x: g4x_wm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1544 {
    mutex: mutex,
    work: delayed_work,
    refresh_rate: u32,
    frontbuffer_bits: u32,
    busy_frontbuffer_bits: u32,
    cpu_transcoder: i32,
    m_n: intel_link_m_n,
    m2_n2: intel_link_m_n,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1554 {
    flip_count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1560 {
    start_vbl_count: u32,
    start_vbl_time: i64,
    min_vbl: i32,
    max_vbl: i32,
    scanline_start: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1588 {
    enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_hprime {
    msg_id: u8,
    h_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1683 {
    r#type: u32,
    max_tmds_clock: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_pairing_info {
    msg_id: u8,
    e_kh_km: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_lc_init {
    msg_id: u8,
    r_n: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_lc_send_lprime {
    msg_id: u8,
    l_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ske_send_eks {
    msg_id: u8,
    e_dkey_ks: [u8; 16],
    riv: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_send_receiverid_list {
    msg_id: u8,
    rx_info: [u8; 2],
    seq_num_v: [u8; 3],
    v_prime: [u8; 16],
    receiver_ids: [u8; 155],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1728 {
    vlv_pps_pipe: i32,
    pps_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_send_ack {
    msg_id: u8,
    v: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1848 {
    active: bool,
    mst_probed_lane_count: i32,
    mst_probed_rate: i32,
    training: *mut intel_dp_link_training,
    caps: *mut intel_dp_link_caps,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1885 {
    stream_encoders: [*mut intel_dp_mst_encoder; 4],
    mgr: drm_dp_mst_topology_mgr,
    active_streams: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1919 {
    min_tmds_clock: i32,
    max_tmds_clock: i32,
    max_dotclock: i32,
    pcon_max_frl_bw: i32,
    max_bpc: u8,
    ycbcr_444_to_420: bool,
    ycbcr420_passthrough: bool,
    rgb_to_ycbcr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1953 {
    transcoder: i32,
    lock: mutex,
    lobf_disable_debug: bool,
    sink_alpm_error: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1963 {
    mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_stream_ready {
    msg_id: u8,
    m_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_2000 {
    mutex: mutex,
    num_streams: u32,
    auth_status: bool,
    port_data: hdcp_port_data,
    mst_type1_capable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp_port_data {
    hdcp_ddi: u32,
    hdcp_transcoder: u32,
    port_type: u8,
    protocol: u8,
    k: u16,
    seq_num_m: u32,
    streams: *mut hdcp2_streamid_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_hdcp_ops {
    owner: *mut module,
    initiate_hdcp2_session: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_init) -> i32>,
    verify_receiver_cert_prepare_km: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_cert, *mut bool, *mut hdcp2_ake_no_stored_km, *mut u64) -> i32>,
    verify_hprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_hprime) -> i32>,
    store_pairing_info: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_pairing_info) -> i32>,
    initiate_locality_check: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_lc_init) -> i32>,
    verify_lprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_lc_send_lprime) -> i32>,
    get_session_key: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ske_send_eks) -> i32>,
    repeater_check_flow_prepare_ack: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_rep_send_receiverid_list, *mut hdcp2_rep_send_ack) -> i32>,
    verify_mprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_rep_stream_ready) -> i32>,
    enable_hdcp_authentication: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data) -> i32>,
    close_hdcp_session: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dsi_h_56 {
    ports: u16,
    phys: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_262 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_262_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_263_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_264_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_264_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_264_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_264_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_264_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_264_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_271 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_271_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_272 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_273_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_274_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_274_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_274_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_274_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_274_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_274_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_281 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_281_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_282 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_283_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_284_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_284_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_284_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_284_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_284_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_284_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_394_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_394_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_394_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_394_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_394_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_394_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_450 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_453 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_453_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_459_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_461 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_462_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_462_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_462_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_462_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_462_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_462_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_469_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_473 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_474_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_474_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_474_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_474_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_474_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_474_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_492 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_494 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_504 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_509 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_511 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_520 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_522 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_534 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_537 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_581 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_596 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_817 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_830 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_831_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_831_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_831_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_831_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_831_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_831_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_955_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_979_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_993_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1008_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1038_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1053 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1057 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1258 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1259 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1277 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1278_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1278_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1278_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1278_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1278_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1278_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1322 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1326 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1386 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1400 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1403 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1610 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_icl_dsi_c_1759 {
}


#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits641U0 {
        #[bits(1)]
        pub for_kupdate: u32,
        #[bits(1)]
        pub for_background: u32,
        #[bits(1)]
        pub tagged_writepages: u32,
        #[bits(1)]
        pub range_cyclic: u32,
        #[bits(1)]
        pub for_sync: u32,
        #[bits(1)]
        pub unpinned_netfs_wb: u32,
        #[bits(1)]
        pub no_cgroup_owner: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits672U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits73U0 {
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
    pub struct __SlateBits73U1 {
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
    pub struct __SlateBits433U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits500U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1193U0 {
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub implicit_on_dfl: bool,
        #[bits(1)]
        pub threaded: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits73U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits124U0 {
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
    pub struct __SlateBits132U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits527U0 {
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
    pub struct __SlateBits524U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits527U1 {
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
    pub struct __SlateBits527U2 {
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
    pub struct __SlateBits527U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits502U0 {
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
    pub struct __SlateBits131U0 {
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
    pub struct __SlateBits881U0 {
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
    pub struct __SlateBits929U0 {
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
    pub struct __SlateBits917U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits733U0 {
        #[bits(18)]
        pub lineno: u32,
        #[bits(6)]
        pub class_id: u32,
        #[bits(8)]
        pub flags: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1237U0 {
        #[bits(1)]
        pub lsb_reg_used: bool,
        #[bits(1)]
        pub aux_enable: bool,
        #[bits(1)]
        pub aux_set: bool,
        #[bits(1)]
        pub luminance_set: bool,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1318U0 {
        #[bits(1)]
        pub lvds_dither: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1297U0 {
        #[bits(1)]
        pub enable_dithering: u32,
        #[bits(1)]
        pub rsvd1: u32,
        #[bits(1)]
        pub is_bridge: u32,
        #[bits(2)]
        pub panel_arch_type: u32,
        #[bits(1)]
        pub is_cmd_mode: u32,
        #[bits(2)]
        pub video_transfer_mode: u32,
        #[bits(1)]
        pub cabc_supported: u32,
        #[bits(1)]
        pub pwm_blc: u32,
        #[bits(4)]
        pub videomode_color_format: u32,
        #[bits(2)]
        pub rotation: u32,
        #[bits(1)]
        pub bta_disable: u32,
        #[bits(15)]
        pub rsvd2: u32,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1298U0 {
        #[bits(2)]
        pub dual_link: u16,
        #[bits(2)]
        pub lane_cnt: u16,
        #[bits(3)]
        pub pixel_overlap: u16,
        #[bits(1)]
        pub rgb_flip: u16,
        #[bits(2)]
        pub dl_dcs_cabc_ports: u16,
        #[bits(2)]
        pub dl_dcs_backlight_ports: u16,
        #[bits(1)]
        pub port_sync: u16,
        #[bits(3)]
        pub rsvd3: u16,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1299U0 {
        #[bits(1)]
        pub dsi_usage: u16,
        #[bits(15)]
        pub rsvd4: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1300U0 {
        #[bits(2)]
        pub byte_clk_sel: u8,
        #[bits(6)]
        pub rsvd6: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1301U0 {
        #[bits(1)]
        pub dphy_param_valid: u16,
        #[bits(1)]
        pub eot_pkt_disabled: u16,
        #[bits(1)]
        pub enable_clk_stop: u16,
        #[bits(1)]
        pub blanking_packets_during_bllp: u16,
        #[bits(1)]
        pub lp_clock_during_lpm: u16,
        #[bits(11)]
        pub rsvd7: u16,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1302U0 {
        #[bits(6)]
        pub prepare_cnt: u32,
        #[bits(2)]
        pub rsvd8: u32,
        #[bits(8)]
        pub clk_zero_cnt: u32,
        #[bits(5)]
        pub trail_cnt: u32,
        #[bits(3)]
        pub rsvd9: u32,
        #[bits(6)]
        pub exit_zero_cnt: u32,
        #[bits(2)]
        pub rsvd10: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1329U0 {
        #[bits(1)]
        pub dsc_hblank_expansion_quirk: u8,
        #[bits(1)]
        pub dsc_throughput_quirk: u8,
        #[bits(1)]
        pub dsc_decompression_enabled: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1248U0 {
        #[bits(1)]
        pub mst_state: bool,
        #[bits(1)]
        pub payload_id_table_cleared: bool,
        #[bits(1)]
        pub reset_rx_state: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits959U0 {
        #[bits(1)]
        pub tunnel_suspended: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 9], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits972U0 {
        #[bits(1)]
        pub g4x: u64,
        #[bits(1)]
        pub mobile: u64,
        #[bits(1)]
        pub dgfx: u64,
        #[bits(1)]
        pub i830: u64,
        #[bits(1)]
        pub i845g: u64,
        #[bits(1)]
        pub i85x: u64,
        #[bits(1)]
        pub i865g: u64,
        #[bits(1)]
        pub i915g: u64,
        #[bits(1)]
        pub i915gm: u64,
        #[bits(1)]
        pub i945g: u64,
        #[bits(1)]
        pub i945gm: u64,
        #[bits(1)]
        pub g33: u64,
        #[bits(1)]
        pub pineview: u64,
        #[bits(1)]
        pub i965g: u64,
        #[bits(1)]
        pub i965gm: u64,
        #[bits(1)]
        pub g45: u64,
        #[bits(1)]
        pub gm45: u64,
        #[bits(1)]
        pub ironlake: u64,
        #[bits(1)]
        pub sandybridge: u64,
        #[bits(1)]
        pub ivybridge: u64,
        #[bits(1)]
        pub valleyview: u64,
        #[bits(1)]
        pub haswell: u64,
        #[bits(1)]
        pub haswell_ult: u64,
        #[bits(1)]
        pub haswell_ulx: u64,
        #[bits(1)]
        pub broadwell: u64,
        #[bits(1)]
        pub broadwell_ult: u64,
        #[bits(1)]
        pub broadwell_ulx: u64,
        #[bits(1)]
        pub cherryview: u64,
        #[bits(1)]
        pub skylake: u64,
        #[bits(1)]
        pub skylake_ult: u64,
        #[bits(1)]
        pub skylake_ulx: u64,
        #[bits(1)]
        pub broxton: u64,
        #[bits(1)]
        pub kabylake: u64,
        #[bits(1)]
        pub kabylake_ult: u64,
        #[bits(1)]
        pub kabylake_ulx: u64,
        #[bits(1)]
        pub geminilake: u64,
        #[bits(1)]
        pub coffeelake: u64,
        #[bits(1)]
        pub coffeelake_ult: u64,
        #[bits(1)]
        pub coffeelake_ulx: u64,
        #[bits(1)]
        pub cometlake: u64,
        #[bits(1)]
        pub cometlake_ult: u64,
        #[bits(1)]
        pub cometlake_ulx: u64,
        #[bits(1)]
        pub icelake: u64,
        #[bits(1)]
        pub icelake_port_f: u64,
        #[bits(1)]
        pub jasperlake: u64,
        #[bits(1)]
        pub elkhartlake: u64,
        #[bits(1)]
        pub tigerlake: u64,
        #[bits(1)]
        pub tigerlake_uy: u64,
        #[bits(1)]
        pub rocketlake: u64,
        #[bits(1)]
        pub dg1: u64,
        #[bits(1)]
        pub alderlake_s: u64,
        #[bits(1)]
        pub alderlake_s_raptorlake_s: u64,
        #[bits(1)]
        pub alderlake_p: u64,
        #[bits(1)]
        pub alderlake_p_alderlake_n: u64,
        #[bits(1)]
        pub alderlake_p_raptorlake_p: u64,
        #[bits(1)]
        pub alderlake_p_raptorlake_u: u64,
        #[bits(1)]
        pub dg2: u64,
        #[bits(1)]
        pub dg2_g10: u64,
        #[bits(1)]
        pub dg2_g11: u64,
        #[bits(1)]
        pub dg2_g12: u64,
        #[bits(1)]
        pub meteorlake: u64,
        #[bits(1)]
        pub meteorlake_u: u64,
        #[bits(1)]
        pub lunarlake: u64,
        #[bits(1)]
        pub battlemage: u64,
        #[bits(1)]
        pub pantherlake: u64,
        #[bits(1)]
        pub pantherlake_wildcatlake: u64,
        #[bits(1)]
        pub novalake: u64,
        #[bits(5, access = na)]
        pub __slate_pad_67: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits975U0 {
        #[bits(1)]
        pub cursor_needs_physical: u8,
        #[bits(1)]
        pub has_cdclk_crawl: u8,
        #[bits(1)]
        pub has_cdclk_squash: u8,
        #[bits(1)]
        pub has_ddi: u8,
        #[bits(1)]
        pub has_dp_mst: u8,
        #[bits(1)]
        pub has_dsb: u8,
        #[bits(1)]
        pub has_fpga_dbg: u8,
        #[bits(1)]
        pub has_gmch: u8,
        #[bits(1)]
        pub has_hotplug: u8,
        #[bits(1)]
        pub has_hti: u8,
        #[bits(1)]
        pub has_ipc: u8,
        #[bits(1)]
        pub has_overlay: u8,
        #[bits(1)]
        pub has_psr: u8,
        #[bits(1)]
        pub has_psr_hw_tracking: u8,
        #[bits(1)]
        pub overlay_needs_physical: u8,
        #[bits(1)]
        pub supports_tv: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1071U0 {
        #[bits(1)]
        pub int_tv_support: u32,
        #[bits(1)]
        pub int_crt_support: u32,
        #[bits(1)]
        pub lvds_use_ssc: u32,
        #[bits(1)]
        pub int_lvds_support: u32,
        #[bits(1)]
        pub display_clock_mode: u32,
        #[bits(1)]
        pub fdi_rx_polarity_inverted: u32,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits463U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1016U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits330U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits330U1 {
        #[bits(5)]
        pub pme_support: u32,
        #[bits(1)]
        pub pme_poll: u32,
        #[bits(1)]
        pub pinned: u32,
        #[bits(1)]
        pub config_rrs_sv: u32,
        #[bits(1)]
        pub imm_ready: u32,
        #[bits(1)]
        pub d1_support: u32,
        #[bits(1)]
        pub d2_support: u32,
        #[bits(1)]
        pub no_d1d2: u32,
        #[bits(1)]
        pub no_d3cold: u32,
        #[bits(1)]
        pub bridge_d3: u32,
        #[bits(1)]
        pub d3cold_allowed: u32,
        #[bits(1)]
        pub mmio_always_on: u32,
        #[bits(1)]
        pub wakeup_prepared: u32,
        #[bits(1)]
        pub skip_bus_pm: u32,
        #[bits(1)]
        pub ignore_hotplug: u32,
        #[bits(1)]
        pub hotplug_user_indicators: u32,
        #[bits(1)]
        pub clear_retrain_link: u32,
        #[bits(1)]
        pub no_bw_notif: u32,
        #[bits(2, access = na)]
        pub __slate_pad_18: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits330U2 {
        #[bits(1)]
        pub aspm_l0s_support: u32,
        #[bits(1)]
        pub aspm_l1_support: u32,
        #[bits(1)]
        pub ltr_path: u32,
        #[bits(1)]
        pub pasid_no_tlp: u32,
        #[bits(3)]
        pub eetlp_prefix_max: u32,
        #[bits(1, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 6], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits330U3 {
        #[bits(1)]
        pub transparent: u32,
        #[bits(1)]
        pub io_window: u32,
        #[bits(1)]
        pub pref_window: u32,
        #[bits(1)]
        pub pref_64_window: u32,
        #[bits(1)]
        pub multifunction: u32,
        #[bits(1)]
        pub is_busmaster: u32,
        #[bits(1)]
        pub no_msi: u32,
        #[bits(1)]
        pub block_cfg_access: u32,
        #[bits(1)]
        pub broken_parity_status: u32,
        #[bits(2)]
        pub irq_reroute_variant: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub ari_enabled: u32,
        #[bits(1)]
        pub ats_enabled: u32,
        #[bits(1)]
        pub pasid_enabled: u32,
        #[bits(1)]
        pub pri_enabled: u32,
        #[bits(1)]
        pub tph_enabled: u32,
        #[bits(1)]
        pub fm_enabled: u32,
        #[bits(1)]
        pub is_managed: u32,
        #[bits(1)]
        pub is_msi_managed: u32,
        #[bits(1)]
        pub needs_freset: u32,
        #[bits(1)]
        pub state_saved: u32,
        #[bits(1)]
        pub is_physfn: u32,
        #[bits(1)]
        pub is_virtfn: u32,
        #[bits(1)]
        pub is_hotplug_bridge: u32,
        #[bits(1)]
        pub is_pciehp: u32,
        #[bits(1)]
        pub shpc_managed: u32,
        #[bits(1)]
        pub is_thunderbolt: u32,
        #[bits(1)]
        pub is_cxl: u32,
        #[bits(1)]
        pub untrusted: u32,
        #[bits(1)]
        pub external_facing: u32,
        #[bits(1)]
        pub broken_intx_masking: u32,
        #[bits(1)]
        pub io_window_1k: u32,
        #[bits(1)]
        pub irq_managed: u32,
        #[bits(1)]
        pub non_compliant_bars: u32,
        #[bits(1)]
        pub is_probed: u32,
        #[bits(1)]
        pub link_active_reporting: u32,
        #[bits(1)]
        pub no_vf_scan: u32,
        #[bits(1)]
        pub no_command_memory: u32,
        #[bits(1)]
        pub rom_bar_overlap: u32,
        #[bits(1)]
        pub rom_attr_enabled: u32,
        #[bits(1)]
        pub non_mappable_bars: u32,
        #[bits(5, access = na)]
        pub __slate_pad_42: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits330U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct i915_hdcp_arbiter {
    hdcp_dev: *mut device,
    ops: *const i915_hdcp_ops,
    mutex: mutex,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct mipi_config {
    panel_id: u16,
    __slate_anon_1: __SlateRecord_intel_dsi_vbt_defs_h_50,
    __slate_anon_2: __SlateRecord_intel_dsi_vbt_defs_h_84,
    __slate_anon_3: __SlateRecord_intel_dsi_vbt_defs_h_102,
    rsvd5: u8,
    target_burst_mode_freq: u32,
    dsi_ddr_clk: u32,
    bridge_ref_clk: u32,
    __slate_anon_8: __SlateRecord_intel_dsi_vbt_defs_h_113,
    __slate_anon_9: __SlateRecord_intel_dsi_vbt_defs_h_122,
    hs_tx_timeout: u32,
    lp_rx_timeout: u32,
    turn_around_timeout: u32,
    device_reset_timer: u32,
    master_init_timer: u32,
    dbi_bw_timer: u32,
    lp_byte_clk_val: u32,
    __slate_anon_17: __SlateRecord_intel_dsi_vbt_defs_h_140,
    clk_lane_switch_cnt: u32,
    hl_switch_cnt: u32,
    rsvd11: [u32; 6],
    tclk_miss: u8,
    tclk_post: u8,
    rsvd12: u8,
    tclk_pre: u8,
    tclk_prepare: u8,
    tclk_settle: u8,
    tclk_term_enable: u8,
    tclk_trail: u8,
    tclk_prepare_clkzero: u16,
    rsvd13: u8,
    td_term_enable: u8,
    teot: u8,
    ths_exit: u8,
    ths_prepare: u8,
    ths_prepare_hszero: u16,
    rsvd14: u8,
    ths_settle: u8,
    ths_skip: u8,
    ths_trail: u8,
    tinit: u8,
    tlpx: u8,
    rsvd15: [u8; 3],
    panel_enable: u8,
    bl_enable: u8,
    pwm_enable: u8,
    reset_r_n: u8,
    pwr_down_r: u8,
    stdby_r_n: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct mipi_pps_data {
    panel_on_delay: u16,
    bl_enable_delay: u16,
    bl_disable_delay: u16,
    panel_off_delay: u16,
    panel_power_cycle_delay: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_encoder {
    base: drm_encoder,
    r#type: u32,
    port: i32,
    cloneable: u16,
    pipe_mask: u8,
    link_check_work: delayed_work,
    link_check: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    hotplug: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_connector) -> u32>,
    compute_output_type: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> u32>,
    compute_config: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> i32>,
    compute_config_late: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> i32>,
    pre_pll_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    pre_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    post_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    post_pll_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    update_pipe: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    audio_enable: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    audio_disable: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_encoder, *mut i32) -> bool>,
    get_config: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state)>,
    sync_state: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    initial_fastset_check: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state) -> bool>,
    get_power_domains: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state)>,
    suspend: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    suspend_complete: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    shutdown: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    shutdown_complete: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    enable_clock: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    disable_clock: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    is_clock_enabled: Option<unsafe extern "C" fn(*mut intel_encoder) -> bool>,
    port_pll_type: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state) -> u32>,
    get_buf_trans: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *mut i32) -> *const intel_ddi_buf_trans>,
    set_signal_levels: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    hpd_pin: u32,
    power_domain: u32,
    devdata: *const intel_bios_encoder_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_panel_bl_funcs {
    setup: Option<unsafe extern "C" fn(*mut intel_connector, i32) -> i32>,
    get: Option<unsafe extern "C" fn(*mut intel_connector, i32) -> u32>,
    set: Option<unsafe extern "C" fn(*const drm_connector_state, u32)>,
    disable: Option<unsafe extern "C" fn(*const drm_connector_state, u32)>,
    enable: Option<unsafe extern "C" fn(*const intel_crtc_state, *const drm_connector_state, u32)>,
    hz_to_pwm: Option<unsafe extern "C" fn(*mut intel_connector, u32) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pps_delays {
    power_up: u16,
    backlight_on: u16,
    backlight_off: u16,
    power_down: u16,
    power_cycle: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_vbt_panel_data {
    lfp_vbt_mode: *mut drm_display_mode,
    sdvo_lvds_vbt_mode: *mut drm_display_mode,
    panel_type: i32,
    __slate_bits_0: __slate_bits::__SlateBits1318U0,
    bios_lvds_val: u32,
    vrr: bool,
    seamless_drrs_min_refresh_rate: u8,
    drrs_type: u32,
    edp: __SlateRecord_intel_display_types_h_345,
    psr: __SlateRecord_intel_display_types_h_360,
    backlight: __SlateRecord_intel_display_types_h_370,
    dsi: __SlateRecord_intel_display_types_h_382,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_panel {
    base: *mut drm_panel,
    fixed_edid: *const drm_edid,
    fixed_modes: list_head,
    backlight: __SlateRecord_intel_display_types_h_407,
    vbt: intel_vbt_panel_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp {
    shim: *const intel_hdcp_shim,
    mutex: mutex,
    value: u64,
    check_work: delayed_work,
    prop_work: work_struct,
    hdcp_encrypted: bool,
    hdcp2_supported: bool,
    hdcp2_encrypted: bool,
    content_type: u8,
    is_paired: bool,
    is_repeater: bool,
    seq_num_v: u32,
    seq_num_m: u32,
    cp_irq_queue: wait_queue_head,
    cp_irq_count: __SlateRecord_types_h_190,
    cp_irq_count_cached: i32,
    cpu_transcoder: i32,
    stream_transcoder: i32,
    force_hdcp14: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_connector {
    base: drm_connector,
    encoder: *mut intel_encoder,
    acpi_device_id: u32,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_connector) -> bool>,
    sync_state: Option<unsafe extern "C" fn(*mut intel_connector, *const intel_crtc_state)>,
    panel: intel_panel,
    detect_edid: *const drm_edid,
    hotplug_retries: i32,
    polled: u8,
    force_joined_pipes: i32,
    dp: __SlateRecord_intel_display_types_h_559,
    mst: __SlateRecord_intel_display_types_h_602,
    link: __SlateRecord_intel_display_types_h_607,
    modeset_retry_work: work_struct,
    hdcp: intel_hdcp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dpll {
    n: i32,
    m1: i32,
    m2: i32,
    p1: i32,
    p2: i32,
    dot: i32,
    vco: i32,
    m: i32,
    p: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_atomic_state {
    base: drm_atomic_commit,
    wakeref: *mut ref_tracker,
    global_objs: *mut intel_global_objs_state,
    num_global_objs: i32,
    internal: bool,
    dpll_set: bool,
    modeset: bool,
    dpll_state: [intel_dpll_state; 9],
    inherited_dp_tunnels: *mut intel_dp_tunnel_inherited_state,
    skip_intermediate_wm: bool,
    rps_interactive: bool,
    cleanup_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_initial_plane_config {
    fb: *mut drm_framebuffer,
    vma: *mut i915_vma,
    size: i32,
    base: u32,
    rotation: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_scaler {
    mode: u32,
    in_use: bool,
    hscale: i32,
    vscale: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_scaler_state {
    scalers: [intel_scaler; 2],
    scaler_users: u32,
    scaler_id: i32,
}


// TE0 or TE1 flag is set if the crtc has a DSI encoder which
// is operating in command mode.
// Flag to use TE from DSI0 instead of VBI in command mode
// Flag to use TE from DSI1 instead of VBI in command mode
// Flag to indicate mipi dsi periodic command mode where we do not get TE
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm_level {
    enable: bool,
    pri_val: u32,
    spr_val: u32,
    cur_val: u32,
    fbc_val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pipe_wm {
    wm: [intel_wm_level; 5],
    fbc_wm_enabled: bool,
    pipe_enabled: bool,
    sprites_enabled: bool,
    sprites_scaled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_wm_level {
    min_ddb_alloc: u16,
    min_ddb_alloc_uv: u16,
    blocks: u16,
    lines: u8,
    enable: bool,
    ignore_lines: bool,
    auto_min_alloc_wm_enable: bool,
    can_sagv: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_plane_wm {
    wm: [skl_wm_level; 8],
    trans_wm: skl_wm_level,
    sagv: __SlateRecord_intel_display_types_h_853,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_pipe_wm {
    planes: [skl_plane_wm; 8],
    use_sagv_wm: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_state {
    wm: [g4x_pipe_wm; 3],
    sr: [g4x_sr_wm; 3],
    num_levels: u8,
    cxsr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_fifo_state {
    plane: [u16; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_wm_state {
    wm: g4x_pipe_wm,
    sr: g4x_sr_wm,
    hpll: g4x_sr_wm,
    cxsr: bool,
    hpll_en: bool,
    fbc_en: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_wm_state {
    __slate_anon_0: __SlateRecord_intel_display_types_h_899,
    need_postvbl_update: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_link_m_n {
    tu: u32,
    data_m: u32,
    data_n: u32,
    link_m: u32,
    link_n: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_csc_matrix {
    coeff: [u16; 9],
    preoff: [u16; 3],
    postoff: [u16; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scaler_filter_coeff {
    sign: u16,
    exp: u16,
    mantissa: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_casf {
    coeff: [scaler_filter_coeff; 7],
    strength: u8,
    win_size: u8,
    enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_state {
    uapi: drm_crtc_state,
    hw: __SlateRecord_intel_display_types_h_1036,
    pre_csc_lut: *mut drm_property_blob,
    post_csc_lut: *mut drm_property_blob,
    csc: intel_csc_matrix,
    output_csc: intel_csc_matrix,
    quirks: u64,
    fb_bits: u32,
    update_pipe: bool,
    update_m_n: bool,
    update_lrr: bool,
    disable_cxsr: bool,
    update_wm_pre: bool,
    update_wm_post: bool,
    fifo_changed: bool,
    preload_luts: bool,
    inherited: bool,
    do_async_flip: bool,
    pipe_src: drm_rect,
    pixel_rate: u32,
    pixel_rate_cdclk: u32,
    has_pch_encoder: bool,
    has_infoframe: bool,
    cpu_transcoder: i32,
    limited_color_range: bool,
    output_types: u32,
    has_hdmi_sink: bool,
    has_audio: bool,
    dither: bool,
    dither_force_disable: bool,
    clock_set: bool,
    sdvo_tv_clock: bool,
    bw_constrained: bool,
    dpll: dpll,
    intel_dpll: *mut intel_dpll,
    dpll_hw_state: intel_dpll_hw_state,
    icl_port_dplls: [icl_port_dpll; 2],
    dsi_pll: __SlateRecord_intel_display_types_h_1171,
    max_link_bpp_x16: i32,
    max_pipe_bpp: i32,
    pipe_bpp: i32,
    min_hblank: i32,
    dp_m_n: intel_link_m_n,
    dp_m2_n2: intel_link_m_n,
    has_drrs: bool,
    has_psr: bool,
    has_sel_update: bool,
    enable_psr2_sel_fetch: bool,
    enable_psr2_su_region_et: bool,
    clear_psr2_sel_fetch: bool,
    req_psr2_sdp_prior_scanline: bool,
    has_panel_replay: bool,
    link_off_after_as_sdp_when_pr_active: bool,
    disable_as_sdp_when_pr_active: bool,
    wm_level_disabled: bool,
    pkg_c_latency_used: bool,
    panel_replay_dsc_support: u32,
    su_y_granularity: u16,
    active_non_psr_pipes: u8,
    entry_setup_frames: u8,
    no_psr_reason: *const u8,
    port_clock: i32,
    pixel_multiplier: u32,
    mode_flags: u8,
    lane_count: u8,
    lane_lat_optim_mask: u8,
    min_voltage_level: u8,
    gmch_pfit: __SlateRecord_intel_display_types_h_1230,
    pch_pfit: __SlateRecord_intel_display_types_h_1237,
    fdi_lanes: i32,
    fdi_m_n: intel_link_m_n,
    ips_enabled: bool,
    crc_enabled: bool,
    double_wide: bool,
    scaler_state: intel_crtc_scaler_state,
    hsw_workaround_pipe: i32,
    wm: intel_crtc_wm_state,
    min_cdclk: i32,
    plane_min_cdclk: [i32; 8],
    data_rate: [u32; 8],
    data_rate_y: [u32; 8],
    rel_data_rate: [u64; 8],
    rel_data_rate_y: [u64; 8],
    gamma_mode: u32,
    __slate_anon_84: __SlateRecord_intel_display_types_h_1277,
    enabled_planes: u8,
    active_planes: u8,
    scaled_planes: u8,
    nv12_planes: u8,
    c8_planes: u8,
    update_planes: u8,
    async_flip_planes: u8,
    framestart_delay: u8,
    msa_timing_delay: u8,
    infoframes: __SlateRecord_intel_display_types_h_1303,
    eld: [u8; 128],
    hdmi_scrambling: bool,
    hdmi_high_tmds_clock_ratio: bool,
    output_format: u32,
    sink_format: u32,
    gamma_enable: bool,
    csc_enable: bool,
    wgc_enable: bool,
    joiner_pipes: u8,
    dsc: __SlateRecord_intel_display_types_h_1347,
    dp_tunnel_ref: drm_dp_tunnel_ref,
    linetime: u16,
    ips_linetime: u16,
    enhanced_framing: bool,
    fec_enable: bool,
    sdp_split_enable: bool,
    master_transcoder: i32,
    sync_mode_slaves_mask: u16,
    mst_master_transcoder: i32,
    dsb_color: *mut intel_dsb,
    dsb_commit: *mut intel_dsb,
    use_dsb: bool,
    use_flipq: bool,
    psr2_man_track_ctl: u32,
    pipe_srcsz_early_tpt: u32,
    psr2_su_area: drm_rect,
    vrr: __SlateRecord_intel_display_types_h_1401,
    cmrr: __SlateRecord_intel_display_types_h_1416,
    splitter: __SlateRecord_intel_display_types_h_1422,
    vblank_work: drm_vblank_work,
    has_lobf: bool,
    set_context_latency: u16,
    alpm_state: __SlateRecord_intel_display_types_h_1437,
    plane_color_changed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pipe_crc {
    lock: spinlock,
    skipped: i32,
    source: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_flipq {
    start_mmioaddr: u32,
    flipq_id: u32,
    tail: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc {
    base: drm_crtc,
    pipe_head: list_head,
    pipe: i32,
    active: bool,
    plane_ids_mask: u8,
    mode_flags: u8,
    vmax_vblank_start: u16,
    enabled_power_domains: intel_display_power_domain_set,
    hw_readout_power_domains: intel_display_power_domain_set,
    overlay: *mut intel_overlay,
    config: *mut intel_crtc_state,
    flip_done_event: *mut drm_pending_vblank_event,
    dsb_event: *mut drm_pending_vblank_event,
    flipq_event: *mut drm_pending_vblank_event,
    cpu_fifo_underrun_disabled: bool,
    pch_fifo_underrun_disabled: bool,
    flipq: [intel_flipq; 5],
    wm: __SlateRecord_intel_display_types_h_1535,
    drrs: __SlateRecord_intel_display_types_h_1544,
    dc_balance: __SlateRecord_intel_display_types_h_1554,
    scanline_offset: i32,
    debug: __SlateRecord_intel_display_types_h_1560,
    num_scalers: i32,
    vblank_pm_qos: pm_qos_request,
    pipe_crc: intel_pipe_crc,
    vblank_psr_notify: bool,
    cmtg: __SlateRecord_intel_display_types_h_1588,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdmi {
    hdmi_reg: __SlateRecord_i915_reg_defs_h_14,
    dp_dual_mode: __SlateRecord_intel_display_types_h_1683,
    attached_connector: *mut intel_connector,
    cec_notifier: *mut cec_notifier,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_compliance_data {
    edid: u64,
    video_pattern: u8,
    hdisplay: u16,
    vdisplay: u16,
    bpc: u8,
    phytest: drm_dp_phy_test_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_compliance {
    test_type: u64,
    test_data: intel_dp_compliance_data,
    test_active: bool,
    test_link_rate: i32,
    test_lane_count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_pcon_frl {
    is_trained: bool,
    trained_rate_gbps: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pps {
    panel_power_up_delay: i32,
    panel_power_down_delay: i32,
    panel_power_cycle_delay: i32,
    backlight_on_delay: i32,
    backlight_off_delay: i32,
    panel_vdd_work: delayed_work,
    want_panel_vdd: bool,
    initializing: bool,
    last_power_on: u64,
    last_backlight_off: u64,
    panel_power_off_time: i64,
    vdd_wakeref: *mut ref_tracker,
    __slate_anon_12: __SlateRecord_intel_display_types_h_1728,
    vlv_active_pipe: i32,
    bxt_pps_reset: bool,
    pps_delays: intel_pps_delays,
    bios_pps_delays: intel_pps_delays,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_psr {
    lock: mutex,
    debug: u32,
    sink_support: bool,
    source_support: bool,
    enabled: bool,
    pause_counter: i32,
    pipe: i32,
    transcoder: i32,
    active: bool,
    work: work_struct,
    busy_frontbuffer_bits: u32,
    link_standby: bool,
    sel_update_enabled: bool,
    psr2_sel_fetch_enabled: bool,
    psr2_sel_fetch_cff_enabled: bool,
    su_region_et_enabled: bool,
    req_psr2_sdp_prior_scanline: bool,
    last_entry_attempt: i64,
    last_exit: i64,
    sink_not_reliable: bool,
    irq_aux_error: bool,
    dc3co_allowed: bool,
    dc3co_work: delayed_work,
    su_w_granularity: u16,
    su_y_granularity: u16,
    source_panel_replay_support: bool,
    sink_panel_replay_support: bool,
    panel_replay_enabled: bool,
    dc3co_exit_delay: u32,
    entry_setup_frames: u8,
    io_wake_lines: u8,
    fast_wake_lines: u8,
    link_ok: bool,
    pkg_c_latency_used: bool,
    active_non_psr_pipes: u8,
    no_psr_reason: *const u8,
    vblank_wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp {
    output_reg: __SlateRecord_i915_reg_defs_h_14,
    DP: u32,
    link_rate: i32,
    lane_count: u8,
    sink_count: u8,
    downstream_port_changed: bool,
    needs_modeset_retry: bool,
    use_max_params: bool,
    dpcd: [u8; 15],
    downstream_ports: [u8; 16],
    edp_dpcd: [u8; 5],
    lttpr_common_caps: [u8; 8],
    lttpr_phy_caps: [[u8; 3]; 8],
    pcon_dsc_dpcd: [u8; 13],
    num_source_rates: i32,
    source_rates: *const i32,
    num_sink_rates: i32,
    sink_rates: [i32; 8],
    use_rate_select: bool,
    max_sink_lane_count: i32,
    link: __SlateRecord_intel_display_types_h_1848,
    reset_link_params: bool,
    mso_link_count: i32,
    mso_pixel_overlap: i32,
    desc: drm_dp_desc,
    aux: drm_dp_aux,
    aux_busy_last_status: u32,
    train_set: [u8; 4],
    pps: intel_pps,
    is_mst: bool,
    mst_detect: u32,
    attached_connector: *mut intel_connector,
    as_sdp_supported: bool,
    as_sdp_v2_supported: bool,
    tunnel: *mut drm_dp_tunnel,
    __slate_bits_0: __slate_bits::__SlateBits959U0,
    disabled_uhbr_lane_mask: u8,
    mst: __SlateRecord_intel_display_types_h_1885,
    get_aux_clock_divider: Option<unsafe extern "C" fn(*mut intel_dp, i32) -> u32>,
    get_aux_send_ctl: Option<unsafe extern "C" fn(*mut intel_dp, i32, u32) -> u32>,
    aux_ch_ctl_reg: Option<unsafe extern "C" fn(*mut intel_dp) -> __SlateRecord_i915_reg_defs_h_14>,
    aux_ch_data_reg: Option<unsafe extern "C" fn(*mut intel_dp, i32) -> __SlateRecord_i915_reg_defs_h_14>,
    prepare_link_retrain: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state)>,
    set_link_train: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state, u8)>,
    set_idle_link_train: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state)>,
    preemph_max: Option<unsafe extern "C" fn(*mut intel_dp) -> u8>,
    voltage_max: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state) -> u8>,
    compliance: intel_dp_compliance,
    dfp: __SlateRecord_intel_display_types_h_1919,
    pm_qos: pm_qos_request,
    force_dsc_en: bool,
    force_dsc_output_format: i32,
    force_dsc_fractional_bpp_en: bool,
    force_dsc_bpc: i32,
    hobl_failed: bool,
    hobl_active: bool,
    frl: intel_dp_pcon_frl,
    psr: intel_psr,
    last_oui_write: u64,
    oui_valid: bool,
    colorimetry_support: bool,
    sst_split_sdp_support: bool,
    alpm: __SlateRecord_intel_display_types_h_1953,
    alpm_dpcd: u8,
    quirks: __SlateRecord_intel_display_types_h_1963,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lspcon {
    active: bool,
    hdr_supported: bool,
    mode: u32,
    vendor: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_digital_port {
    base: intel_encoder,
    dp: intel_dp,
    hdmi: intel_hdmi,
    lspcon: intel_lspcon,
    hpd_pulse: Option<unsafe extern "C" fn(*mut intel_digital_port, bool) -> u32>,
    lane_reversal: bool,
    ddi_a_4_lanes: bool,
    release_cl2_override: bool,
    dedicated_external: bool,
    max_lanes: u8,
    aux_ch: i32,
    ddi_io_power_domain: u32,
    ddi_io_wakeref: *mut ref_tracker,
    aux_wakeref: *mut ref_tracker,
    tc: *mut intel_tc_port,
    hdcp: __SlateRecord_intel_display_types_h_2000,
    write_infoframe: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, u32, *const (), i64)>,
    read_infoframe: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, u32, *mut (), i64)>,
    set_infoframes: Option<unsafe extern "C" fn(*mut intel_encoder, bool, *const intel_crtc_state, *const drm_connector_state)>,
    infoframes_enabled: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state) -> u32>,
    connected: Option<unsafe extern "C" fn(*mut intel_encoder) -> bool>,
    lock: Option<unsafe extern "C" fn(*mut intel_digital_port)>,
    unlock: Option<unsafe extern "C" fn(*mut intel_digital_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_mst_encoder {
    base: intel_encoder,
    pipe: i32,
    primary: *mut intel_digital_port,
    connector: *mut intel_connector,
}


fn dp_to_dig_port(intel_dp: *mut intel_dp) -> *mut intel_digital_port {
container_of!(intel_dp, intel_digital_port)
}

fn hdmi_to_dig_port(intel_hdmi: *mut intel_hdmi) -> *mut intel_digital_port {
container_of__2!(intel_hdmi, intel_digital_port)
}

// Helper for generic association. Map types to conversion functions/macros.
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsi {
    base: intel_encoder,
    dsi_hosts: [*mut intel_dsi_host; 9],
    io_wakeref: [*mut ref_tracker; 9],
    gpio_panel: *mut gpio_desc,
    gpio_backlight: *mut gpio_desc,
    attached_connector: *mut intel_connector,
    __slate_anon_6: __SlateRecord_intel_dsi_h_56,
    channel: i32,
    operation_mode: u16,
    lane_count: u32,
    i2c_bus_num: i32,
    pixel_format: u32,
    video_mode: i32,
    lp_clock_during_lpm: bool,
    blanking_pkt: bool,
    eot_pkt: bool,
    clock_stop: bool,
    escape_clk_div: u8,
    dual_link: u8,
    bgr_enabled: bool,
    pixel_overlap: u8,
    bw_timer: u32,
    dphy_reg: u32,
    dphy_data_lane_reg: u32,
    video_frmt_cfg_bits: u32,
    lp_byte_clk: u16,
    hs_tx_timeout: u16,
    lp_rx_timeout: u16,
    turn_arnd_val: u16,
    rst_timer_val: u16,
    hs_to_lp_count: u16,
    clk_lp_to_hs_count: u16,
    clk_hs_to_lp_count: u16,
    init_count: u16,
    pclk: u32,
    burst_mode_ratio: u16,
    backlight_off_delay: u16,
    backlight_on_delay: u16,
    panel_on_delay: u16,
    panel_off_delay: u16,
    panel_pwr_cycle_delay: u16,
    panel_power_off_time: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsi_host {
    base: mipi_dsi_host,
    intel_dsi: *mut intel_dsi,
    port: i32,
    device: *mut mipi_dsi_device,
}


fn to_intel_dsi_host(h: *mut mipi_dsi_host) -> *mut intel_dsi_host {
container_of__4!(h, intel_dsi_host)
}

fn enc_to_intel_dsi(encoder: *mut intel_encoder) -> *mut intel_dsi {
let mut __v0: *mut intel_dsi;
__v0 = ((unsafe { ((core::ptr::addr_of_mut!((*encoder).base) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!(intel_dsi) as isize)) }) as *mut ()) as *mut intel_dsi;
__v0
}

fn is_vid_mode(intel_dsi: *mut intel_dsi) -> bool {
((unsafe { (*intel_dsi).operation_mode }) as i32) == 0
}

fn is_cmd_mode(intel_dsi: *mut intel_dsi) -> bool {
((unsafe { (*intel_dsi).operation_mode }) as i32) == 1
}

fn header_credits_available(display: *mut intel_display, dsi_trans: i32) -> i32 {
let mut __slate_compound_0: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
((intel_de_read(display, unsafe { *{
__slate_compound_0 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438480i32, 440528i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }) & 7936u32) >> 8i32) as i32
}

fn payload_credits_available(display: *mut intel_display, dsi_trans: i32) -> i32 {
let mut __slate_compound_0: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
((intel_de_read(display, unsafe { *{
__slate_compound_0 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438480i32, 440528i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} }) & 255u32) >> 0i32) as i32
}

fn wait_for_header_credits(display: *mut intel_display, dsi_trans: i32, hdr_credit: i32) -> bool {
let mut ret: i32 = 0;
let mut available: i32 = 0;
let mut __v0: i32;
let __timeout_us: u64 = 100;
let __sleep_us: u64 = 10;
let __timeout: i64 = ktime_add_us(unsafe { ktime_get() }, __timeout_us);
let mut ___ret: i32 = 0;
might_sleep_if!(__sleep_us != 0);
if false && __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
loop {
let mut __expired: bool = false;
__expired = __timeout_us != 0 && ktime_compare(unsafe { ktime_get() }, __timeout) > 0;
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
available = header_credits_available(display, dsi_trans);
if available >= hdr_credit {
___ret = 0;
break;
}
if __expired {
___ret = 110i32.wrapping_neg();
break;
}
if __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
cpu_relax();
}
__v0 = ___ret;
ret = __v0;
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI header credits not released\n".as_ptr() as *const u8) };
return false;
}
true
}

fn wait_for_payload_credits(display: *mut intel_display, dsi_trans: i32, payld_credit: i32) -> bool {
let mut ret: i32 = 0;
let mut available: i32 = 0;
let mut __v0: i32;
let __timeout_us: u64 = 100;
let __sleep_us: u64 = 10;
let __timeout: i64 = ktime_add_us(unsafe { ktime_get() }, __timeout_us);
let mut ___ret: i32 = 0;
might_sleep_if!(__sleep_us != 0);
if false && __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
loop {
let mut __expired: bool = false;
__expired = __timeout_us != 0 && ktime_compare(unsafe { ktime_get() }, __timeout) > 0;
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
available = payload_credits_available(display, dsi_trans);
if available >= payld_credit {
___ret = 0;
break;
}
if __expired {
___ret = 110i32.wrapping_neg();
break;
}
if __sleep_us != 0 {
usleep_range((__sleep_us >> 2i32).wrapping_add(1), __sleep_us);
}
cpu_relax();
}
__v0 = ___ret;
ret = __v0;
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI payload credits not released\n".as_ptr() as *const u8) };
return false;
}
true
}

fn dsi_port_to_transcoder(port: i32) -> i32 {
if (port as i32) == 0 {
return 5;
} else {
return 6;
}
}

fn wait_for_cmds_dispatched_to_panel(encoder: *mut intel_encoder) {
let mut __slate_compound_14: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
// wait for header/payload credits to be released
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
let mut ret: i32 = 0;
while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
wait_for_header_credits(display, dsi_trans, 16i32);
wait_for_payload_credits(display, dsi_trans, 64i32);
}
port = (port as i32).wrapping_add(1) as i32;
}
// send nop DCS command
port = 0;
while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
let dsi: *mut mipi_dsi_device = unsafe { (*(*((*intel_dsi).dsi_hosts.as_mut_ptr() as *mut *mut intel_dsi_host).wrapping_offset((port as i32) as isize))).device };
let __v5: u64 = unsafe { (*dsi).mode_flags };
let __v6: u64 = __v5 | BIT!(11i32);
unsafe {
(*dsi).mode_flags = __v6;
}
unsafe {
(*dsi).channel = 0;
}
ret = unsafe { mipi_dsi_dcs_nop(dsi) };
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* error sending DCS NOP command\n".as_ptr() as *const u8) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
// wait for header credits to be released
port = 0;
while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
wait_for_header_credits(display, dsi_trans, 16i32);
}
port = (port as i32).wrapping_add(1) as i32;
}
// wait for LP TX in progress bit to be cleared
port = 0;
'__slate_break_16: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
ret = unsafe { intel_de_wait_for_clear_us(display, *{
__slate_compound_14 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438488i32, 440536i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_14)
}, 131072u32, 20u32) };
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* LPTX bit not cleared\n".as_ptr() as *const u8) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
}

fn dsi_send_pkt_payld(host: *mut intel_dsi_host, packet: &mipi_dsi_packet) -> i32 {
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let intel_dsi: *mut intel_dsi = unsafe { (*host).intel_dsi };
let display: *mut intel_display = to_intel_display!(unsafe { core::ptr::addr_of_mut!((*intel_dsi).base) });
let dsi_trans: i32 = dsi_port_to_transcoder(unsafe { (*host).port });
let mut data: *const u8 = unsafe { (*(packet as *const mipi_dsi_packet)).payload };
let len: u32 = (unsafe { (*(packet as *const mipi_dsi_packet)).payload_length }) as u32;
let mut i: i32 = 0;
let mut j: i32 = 0;
// payload queue can accept *256 bytes*, check limit
if len > 256u32 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* payload size exceeds max queue limit\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
'__slate_break_14: while (i as u32) < len {
let mut tmp: u32 = 0;
if !wait_for_payload_credits(display, dsi_trans, 1i32) {
return 16i32.wrapping_neg();
}
j = 0;
while (j as u32) < min_t__2!(u32, len.wrapping_sub(i as u32), 4i32) {
let __v2: *const u8 = data;
data = __v2.wrapping_offset(1 as isize);
let __v4: i32 = ((unsafe { *__v2 }) as i32) << 8i32.wrapping_mul(j);
tmp |= __v4 as u32;
j = j.wrapping_add(1);
}
intel_de_write(display, unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438532i32, 440580i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} }, tmp);
i = i.wrapping_add(4);
}
0
}

fn dsi_send_pkt_hdr(host: *mut intel_dsi_host, packet: &mipi_dsi_packet, enable_lpdt: bool) -> i32 {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let intel_dsi: *mut intel_dsi = unsafe { (*host).intel_dsi };
let display: *mut intel_display = to_intel_display!(unsafe { core::ptr::addr_of_mut!((*intel_dsi).base) });
let dsi_trans: i32 = dsi_port_to_transcoder(unsafe { (*host).port });
let mut tmp: u32 = 0;
if !wait_for_header_credits(display, dsi_trans, 1i32) {
return 16i32.wrapping_neg();
}
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438528i32, 440576i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
if !unsafe { (*(packet as *const mipi_dsi_packet)).payload }.is_null() {
tmp |= 2147483648u32;
} else {
let __v5: u32 = tmp & 2147483647u32;
tmp = __v5;
}
let __v7: u32 = tmp & 2415919103u32;
tmp = __v7;
if enable_lpdt {
tmp |= 1073741824u32;
} else {
tmp |= 268435456u32;
}
let __v13: u32 = tmp & 4278190080u32;
tmp = __v13;
let __v14: u32 = tmp;
tmp = __v14 | (((((unsafe { *((*(packet as *const mipi_dsi_packet)).header.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & 192i32) << 6i32) as u32);
let __v16: u32 = tmp;
tmp = __v16 | (((((unsafe { *((*(packet as *const mipi_dsi_packet)).header.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & 63i32) << 0i32) as u32);
let __v18: u32 = tmp;
tmp = __v18 | ((((unsafe { *((*(packet as *const mipi_dsi_packet)).header.as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) << 8i32) as u32);
let __v20: u32 = tmp;
tmp = __v20 | ((((unsafe { *((*(packet as *const mipi_dsi_packet)).header.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) << 16i32) as u32);
intel_de_write(display, unsafe { *{
__slate_compound_22 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438528i32, 440576i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_22)
} }, tmp);
0
}

#[unsafe(no_mangle)]
extern "C" fn icl_dsi_frame_update(crtc_state: *mut intel_crtc_state) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display__5!(crtc_state);
let mode_flags: u32 = (unsafe { (*crtc_state).mode_flags }) as u32;
let mut port: i32 = 0;
// case 1 also covers dual link
// In case of dual link, frame update should be set on
// DSI_0
if mode_flags & 8u32 != 0 {
port = 0;
} else {
if mode_flags & 16u32 != 0 {
port = 1;
} else {
return;
}
}
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438324i32, 440372i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 0u32, 2147483648u32);
}

fn dsi_program_swing_and_deemphasis(encoder: *mut intel_encoder) {
let mut __slate_compound_2: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_20: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut phy: i32 = 0;
let mut tmp: u32 = 0;
let mut mask: u32 = 0;
let mut val: u32 = 0;
let mut lane: i32 = 0;
'__slate_break_28: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
// Program voltage swing and pre-emphasis level values as per
// table in BSPEC under DDI buffer programming.
mask = REG_GENMASK__164!(20i32, 18i32) | REG_GENMASK__211!(5i32, 3i32);
val = SCALING_MODE_SEL!(2i32) | REG_BIT__897!(30i32) | REG_BIT__1161!(29i32) | RTERM_SELECT!(6i32);
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_2 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(0i32) as u32).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
let __v4: u32 = tmp & !mask;
tmp = __v4;
tmp |= val;
intel_de_write(display, unsafe { *{
__slate_compound_7 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_8 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_8)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1664).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_7)
} }, tmp);
intel_de_rmw(display, unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_10 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_10)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_9)
} }, mask, val);
mask = REG_GENMASK__261!(13i32, 11i32) | REG_BIT__264!(15i32) | REG_GENMASK__157!(7i32, 0i32);
val = SWING_SEL_UPPER!(2i32) | SWING_SEL_LOWER!(2i32) | RCOMP_SCALAR__2!(152i32);
tmp = intel_de_read(display, unsafe { *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_12 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_12)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(0i32) as u32).wrapping_add(8u32) };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
let __v14: u32 = tmp & !mask;
tmp = __v14;
tmp |= val;
intel_de_write(display, unsafe { *{
__slate_compound_17 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_18 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_18)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1664).wrapping_add(8u32) };
core::ptr::addr_of_mut!(__slate_compound_17)
} }, tmp);
intel_de_rmw(display, unsafe { *{
__slate_compound_19 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_20 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_20)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(8u32) };
core::ptr::addr_of_mut!(__slate_compound_19)
} }, mask, val);
mask = REG_GENMASK__114!(17i32, 12i32) | REG_GENMASK__230!(11i32, 6i32) | REG_GENMASK__44!(5i32, 0i32);
val = POST_CURSOR_1__2!(0i32) | POST_CURSOR_2!(0i32) | CURSOR_COEFF!(63i32);
intel_de_rmw(display, unsafe { *{
__slate_compound_21 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_22 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_22)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(16u32) };
core::ptr::addr_of_mut!(__slate_compound_21)
} }, mask, val);
// Bspec: must not use GRP register for write
lane = 0;
'__slate_break_25: while lane <= 3 {
intel_de_rmw(display, unsafe { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_24 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_24)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(lane) as u32).wrapping_add(16u32) };
core::ptr::addr_of_mut!(__slate_compound_23)
} }, mask, val);
lane = lane.wrapping_add(1);
}
}
phy = (phy as i32).wrapping_add(1) as i32;
}
}

fn configure_dual_link_mode(encoder: *mut intel_encoder, pipe_config: *const intel_crtc_state) {
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut dss_ctl1_reg: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dss_ctl1_reg) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut dss_ctl2_reg: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dss_ctl2_reg) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut dss_ctl1: u32 = 0;
// FIXME: Move all DSS handling to intel_vdsc.c
if (DISPLAY_VER!(display) as i32) >= 12 {
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*pipe_config).uapi.crtc });
dss_ctl1_reg = unsafe { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PIPE__3!(((*crtc).pipe as i32).wrapping_sub(1), 492032i32, 492544i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} };
dss_ctl2_reg = unsafe { *{
__slate_compound_3 = __SlateRecord_i915_reg_defs_h_14 { reg: _PIPE__3!(((*crtc).pipe as i32).wrapping_sub(1), 492036i32, 492548i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_3)
} };
} else {
dss_ctl1_reg = unsafe { *{
__slate_compound_4 = __SlateRecord_i915_reg_defs_h_14 { reg: 422912 };
core::ptr::addr_of_mut!(__slate_compound_4)
} };
dss_ctl2_reg = unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: 422916 };
core::ptr::addr_of_mut!(__slate_compound_5)
} };
}
dss_ctl1 = intel_de_read(display, dss_ctl1_reg);
dss_ctl1 |= 2147483648u32;
let __v9: u32 = dss_ctl1 & 4293984255u32;
dss_ctl1 = __v9;
let __v10: u32 = dss_ctl1;
dss_ctl1 = __v10 | (OVERLAP_PIXELS!(unsafe { (*intel_dsi).pixel_overlap }) as u32);
if ((unsafe { (*intel_dsi).dual_link }) as i32) == 1 {
let adjusted_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!((*pipe_config).hw.adjusted_mode) };
let hactive: u16 = unsafe { (*adjusted_mode).crtc_hdisplay };
let mut dl_buffer_depth: u16 = 0;
let __v13: u32 = dss_ctl1 & 4278190079u32;
dss_ctl1 = __v13;
dl_buffer_depth = ((hactive as i32) / 2).wrapping_add((unsafe { (*intel_dsi).pixel_overlap }) as i32) as u16;
if (dl_buffer_depth as i32) > 1440 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DL buffer depth exceed max value\n".as_ptr() as *const u8) };
}
let __v16: u32 = dss_ctl1 & 4294963200u32;
dss_ctl1 = __v16;
let __v17: u32 = dss_ctl1;
dss_ctl1 = __v17 | (LEFT_DL_BUF_TARGET_DEPTH!(dl_buffer_depth) as u32);
intel_de_rmw(display, dss_ctl2_reg, 4095u32, RIGHT_DL_BUF_TARGET_DEPTH!(dl_buffer_depth) as u32);
} else {
// Interleave
dss_ctl1 |= 16777216u32;
}
intel_de_write(display, dss_ctl1_reg, dss_ctl1);
}

/// aka DSI 8X clock
fn afe_clk(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) -> i32 {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let bpp: i32 = if unsafe { (*crtc_state).dsc.compression_enable } {
fxp_q4_to_int((unsafe { (*crtc_state).dsc.compressed_bpp_x16 }) as i32)
} else {
mipi_dsi_pixel_format_to_bpp(unsafe { (*intel_dsi).pixel_format })
};
let mut __v0: u32;
let __x: u32 = unsafe { (*intel_dsi).pclk }.wrapping_mul(bpp as u32);
let __d: u32 = unsafe { (*intel_dsi).lane_count };
__v0 = if (1i32.wrapping_neg() as u32) > 0 || (1i32.wrapping_neg() as u32) > 0 || ((__x > 0) as i32) == ((__d > 0) as i32) { __x.wrapping_add(__d / 2) / __d } else { __x.wrapping_sub(__d / 2) / __d };
__v0 as i32
}

fn gen11_dsi_program_esc_clk_div(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_3: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let afe_clk_khz: i32 = afe_clk(encoder, crtc_state);
let mut theo_word_clk: i32 = 0;
let mut act_word_clk: i32 = 0;
let mut esc_clk_div_m: u32 = 0;
let mut esc_clk_div_m_phy: u32 = 0;
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_alderlake_s() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_alderlake_p() }) as i32) != 0 {
theo_word_clk = afe_clk_khz.wrapping_add(160000i32).wrapping_sub(1) / 160000i32;
let mut __v1: i32;
let __UNIQUE_ID_x__706: i32 = 3;
let __UNIQUE_ID_y__707: i32 = theo_word_clk.wrapping_add(theo_word_clk.wrapping_add(1) % 2);
loop {
if __types_ok!(__UNIQUE_ID_x__706, __UNIQUE_ID_y__707) == 0 {
unsafe { __compiletime_assert_708() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__706 > __UNIQUE_ID_y__707 { __UNIQUE_ID_x__706 } else { __UNIQUE_ID_y__707 };
act_word_clk = __v1;
esc_clk_div_m = act_word_clk.wrapping_mul(8) as u32;
esc_clk_div_m_phy = (act_word_clk.wrapping_sub(1) / 2) as u32;
} else {
esc_clk_div_m = (afe_clk_khz.wrapping_add(20000).wrapping_sub(1) / 20000) as u32;
}
'__slate_break_5: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_3 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438416i32, 440464i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_3)
} }, esc_clk_div_m & 511);
intel_de_posting_read(display, unsafe { *{
__slate_compound_4 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438416i32, 440464i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
port = (port as i32).wrapping_add(1) as i32;
}
port = 0;
'__slate_break_10: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_8 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 1450384i32, 442768i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_8)
} }, esc_clk_div_m & 511);
intel_de_posting_read(display, unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 1450384i32, 442768i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
}
port = (port as i32).wrapping_add(1) as i32;
}
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_alderlake_s() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_alderlake_p() }) as i32) != 0 {
port = 0;
'__slate_break_17: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_14 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_14)
}).as_ptr() as *const u32).wrapping_offset((port as i32) as isize) }.wrapping_add(384).wrapping_add(32u32) };
core::ptr::addr_of_mut!(__slate_compound_13)
} }, esc_clk_div_m_phy & (127u32 << (((core::mem::size_of::<__SlateRecord_icl_dsi_c_394_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(23)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_icl_dsi_c_394_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(23))).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_394_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_394_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_394_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_icl_dsi_c_394_16>() as u32));
intel_de_posting_read(display, unsafe { *{
__slate_compound_15 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_16 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_16)
}).as_ptr() as *const u32).wrapping_offset((port as i32) as isize) }.wrapping_add(384).wrapping_add(32u32) };
core::ptr::addr_of_mut!(__slate_compound_15)
} });
}
port = (port as i32).wrapping_add(1) as i32;
}
}
}

fn get_dsi_io_power_domains(intel_dsi: *mut intel_dsi) {
let display: *mut intel_display = to_intel_display!(unsafe { core::ptr::addr_of_mut!((*intel_dsi).base) });
let mut port: i32 = 0;
while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
let mut __v1: i64;
let __ret_warn_on: i32 = !unsafe { *((*intel_dsi).io_wakeref.as_mut_ptr() as *mut *mut ref_tracker).wrapping_offset((port as i32) as isize) }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dsi->io_wakeref[port])".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_get_dsi_io_power_domains_4, sym __slate_asm_string_get_dsi_io_power_domains_5, const 406 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_dsi->io_wakeref[port])".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
unsafe {
*((*intel_dsi).io_wakeref.as_mut_ptr() as *mut *mut ref_tracker).wrapping_offset((port as i32) as isize) = intel_display_power_get(display, (if (port as i32) == 0 { 29i32 } else { 30i32 }) as u32);
}
}
port = (port as i32).wrapping_add(1) as i32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_709: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn gen11_dsi_enable_io_power(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
'__slate_break_2: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438420i32, 440468i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 0u32, 1u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
get_dsi_io_power_domains(intel_dsi);
}

fn gen11_dsi_power_up_lanes(encoder: *mut intel_encoder) {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut phy: i32 = 0;
while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
unsafe { intel_combo_phy_power_up_lanes(display, phy, true, (*intel_dsi).lane_count as i32, false) };
}
phy = (phy as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_config_phy_lanes_sequence(encoder: *mut intel_encoder) {
let mut __slate_compound_2: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_20: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_30: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
// Step 4b(i) set loadgen select for transmit and aux lanes
let mut phy: i32 = 0;
let mut tmp: u32 = 0;
let mut lane: i32 = 0;
'__slate_break_8: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_2 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(16u32) };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_BIT__943!(31i32), 0u32);
lane = 0;
'__slate_break_5: while lane <= 3 {
intel_de_rmw(display, unsafe { *{
__slate_compound_3 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_4 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_4)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(lane) as u32).wrapping_add(16u32) };
core::ptr::addr_of_mut!(__slate_compound_3)
} }, REG_BIT__422!(31i32), if lane != 2 { REG_BIT__1430!(31i32) } else { 0u32 });
lane = lane.wrapping_add(1);
}
}
phy = (phy as i32).wrapping_add(1) as i32;
}
// Step 4b(ii) set latency optimization for transmit and aux lanes
phy = 0;
'__slate_break_31: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_12 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_12)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(8u32) };
core::ptr::addr_of_mut!(__slate_compound_11)
} }, REG_GENMASK__137!(10i32, 8i32), FRC_LATENCY_OPTIM_VAL!(5i32));
tmp = intel_de_read(display, unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_14 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_14)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(0i32) as u32).wrapping_add(8u32) };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
let __v15: u32 = tmp;
tmp = __v15 & !REG_GENMASK__70!(10i32, 8i32);
let __v17: u32 = tmp;
tmp = __v17 | FRC_LATENCY_OPTIM_VAL__2!(5i32);
intel_de_write(display, unsafe { *{
__slate_compound_19 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_20 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_20)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1664).wrapping_add(8u32) };
core::ptr::addr_of_mut!(__slate_compound_19)
} }, tmp);
// For EHL, TGL, set latency optimization for PCS_DW1 lanes
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_jasperlake() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_elkhartlake() }) as i32) != 0 || (DISPLAY_VER!(display) as i32) >= 12 {
intel_de_rmw(display, unsafe { *{
__slate_compound_21 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_22 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_22)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(768).wrapping_add(4u32) };
core::ptr::addr_of_mut!(__slate_compound_21)
} }, REG_GENMASK__24!(3i32, 2i32), LATENCY_OPTIM_VAL__2!(0i32));
tmp = intel_de_read(display, unsafe { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_24 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_24)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_PCS_LN!(0i32) as u32).wrapping_add(4u32) };
core::ptr::addr_of_mut!(__slate_compound_23)
} });
let __v25: u32 = tmp;
tmp = __v25 & !REG_GENMASK__85!(3i32, 2i32);
let __v27: u32 = tmp;
tmp = __v27 | LATENCY_OPTIM_VAL!(1i32);
intel_de_write(display, unsafe { *{
__slate_compound_29 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_30 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_30)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1536).wrapping_add(4u32) };
core::ptr::addr_of_mut!(__slate_compound_29)
} }, tmp);
}
}
phy = (phy as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_voltage_swing_program_seq(encoder: *mut intel_encoder) {
let mut __slate_compound_2: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_28: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_33: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_32: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_35: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_34: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut tmp: u32 = 0;
// clear common keeper enable bit
let mut phy: i32 = 0;
'__slate_break_9: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_2 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_PCS_LN!(0i32) as u32).wrapping_add(4u32) };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
let __v3: u32 = tmp;
tmp = __v3 & !REG_BIT__834!(26i32);
intel_de_write(display, unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_6 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_6)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1536).wrapping_add(4u32) };
core::ptr::addr_of_mut!(__slate_compound_5)
} }, tmp);
intel_de_rmw(display, unsafe { *{
__slate_compound_7 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_8 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_8)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(768).wrapping_add(4u32) };
core::ptr::addr_of_mut!(__slate_compound_7)
} }, REG_BIT__1264!(26i32), 0u32);
}
phy = (phy as i32).wrapping_add(1) as i32;
}
// Set SUS Clock Config bitfield to 11b
// Note: loadgen select program is done
// as part of lane phy sequence configuration
phy = 0;
'__slate_break_14: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_12 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_13 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_13)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_12)
} }, 0u32, REG_GENMASK__163!(1i32, 0i32));
}
phy = (phy as i32).wrapping_add(1) as i32;
}
// Clear training enable to change swing values
phy = 0;
'__slate_break_25: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
tmp = intel_de_read(display, unsafe { *{
__slate_compound_17 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_18 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_18)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(0i32) as u32).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_17)
} });
let __v19: u32 = tmp;
tmp = __v19 & !REG_BIT__1002!(31i32);
intel_de_write(display, unsafe { *{
__slate_compound_21 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_22 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_22)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1664).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_21)
} }, tmp);
intel_de_rmw(display, unsafe { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_24 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_24)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_23)
} }, REG_BIT__1289!(31i32), 0u32);
}
phy = (phy as i32).wrapping_add(1) as i32;
}
// Program swing and de-emphasis
dsi_program_swing_and_deemphasis(encoder);
// Set training enable to trigger update
phy = 0;
'__slate_break_36: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
tmp = intel_de_read(display, unsafe { *{
__slate_compound_28 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_29 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_29)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(_ICL_PORT_TX_LN!(0i32) as u32).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_28)
} });
let __v30: u32 = tmp;
tmp = __v30 | REG_BIT__187!(31i32);
intel_de_write(display, unsafe { *{
__slate_compound_32 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_33 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_33)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(1664).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_32)
} }, tmp);
intel_de_rmw(display, unsafe { *{
__slate_compound_34 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_35 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_35)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(896).wrapping_add(20u32) };
core::ptr::addr_of_mut!(__slate_compound_34)
} }, 0u32, REG_BIT__539!(31i32));
}
phy = (phy as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_enable_ddi_buffer(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let mut ret: i32 = 0;
'__slate_break_4: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 409600i32, 409856i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 0u32, REG_BIT__1052!(31i32));
ret = unsafe { intel_de_wait_for_clear_us(display, *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 409600i32, 409856i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
}, REG_BIT__1373!(7i32), 500u32) };
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DDI port:%c buffer idle\n".as_ptr() as *const u8, port_name!(port)) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_setup_dphy_timings(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
// Program DPHY clock lanes timings
let mut port: i32 = 0;
let mut phy: i32 = 0;
'__slate_break_2: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 1450368i32, 442752i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, unsafe { (*intel_dsi).dphy_reg });
}
port = (port as i32).wrapping_add(1) as i32;
}
// Program DPHY data lanes timings
port = 0;
'__slate_break_6: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 1450372i32, 442756i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_5)
} }, unsafe { (*intel_dsi).dphy_data_lane_reg });
}
port = (port as i32).wrapping_add(1) as i32;
}
// If DSI link operating at or below an 800 MHz,
// TA_SURE should be override and programmed to
// a value '0' inside TA_PARAM_REGISTERS otherwise
// leave all fields at HW default values.
if (DISPLAY_VER!(display) as i32) == 11 {
if afe_clk(encoder, crtc_state) <= 800000 {
port = 0;
'__slate_break_10: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 1450376i32, 442760i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_9)
} }, 2031616u32, (-2147483648i32 | TA_SURE!(0i32)) as u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
}
}
if ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_jasperlake() }) as i32) != 0 || ((unsafe { (*display).platform.__slate_anon_0.__slate_anon_0.__slate_bits_0.__get_elkhartlake() }) as i32) != 0 {
phy = 0;
'__slate_break_15: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: unsafe { *((*{
__slate_compound_14 = [1449984u32, 442368u32, 1441792u32, 1445888u32, 1486848u32];
core::ptr::addr_of_mut!(__slate_compound_14)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }.wrapping_add(404) };
core::ptr::addr_of_mut!(__slate_compound_13)
} }, 0u32, REG_BIT__583!(7i32));
}
phy = (phy as i32).wrapping_add(1) as i32;
}
}
}

fn gen11_dsi_setup_timings(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
// Program T-INIT master registers
let mut port: i32 = 0;
'__slate_break_2: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438408i32, 440456i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_GENMASK__56!(15i32, 0i32), (unsafe { (*intel_dsi).init_count }) as u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
// shadow register inside display core
port = 0;
'__slate_break_6: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438400i32, 440448i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_5)
} }, unsafe { (*intel_dsi).dphy_reg });
}
port = (port as i32).wrapping_add(1) as i32;
}
// shadow register inside display core
port = 0;
'__slate_break_10: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_write(display, unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438404i32, 440452i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_9)
} }, unsafe { (*intel_dsi).dphy_data_lane_reg });
}
port = (port as i32).wrapping_add(1) as i32;
}
// shadow register inside display core
if (DISPLAY_VER!(display) as i32) == 11 {
if afe_clk(encoder, crtc_state) <= 800000 {
port = 0;
'__slate_break_14: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438424i32, 440472i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} }, 2031616u32, (-2147483648i32 | TA_SURE!(0i32)) as u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_gate_clocks")]
extern "C" fn gen11_dsi_gate_clocks(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut tmp: u32 = 0;
let mut phy: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*display).dpll.lock)) };
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
'__slate_break_5: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
let __v2: u32 = tmp;
tmp = __v2 | ((1i32 << unsafe { *((*{
__slate_compound_4 = [10u32, 11u32, 24u32, 4u32, 5u32];
core::ptr::addr_of_mut!(__slate_compound_4)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }) as u32);
}
phy = (phy as i32).wrapping_add(1) as i32;
}
intel_de_write(display, unsafe { *{
__slate_compound_8 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_8)
} }, tmp);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*display).dpll.lock)) };
}

fn gen11_dsi_ungate_clocks(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut tmp: u32 = 0;
let mut phy: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*display).dpll.lock)) };
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
'__slate_break_5: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
let __v2: u32 = tmp;
tmp = __v2 & (!(1i32 << unsafe { *((*{
__slate_compound_4 = [10u32, 11u32, 24u32, 4u32, 5u32];
core::ptr::addr_of_mut!(__slate_compound_4)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }) as u32);
}
phy = (phy as i32).wrapping_add(1) as i32;
}
intel_de_write(display, unsafe { *{
__slate_compound_8 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_8)
} }, tmp);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*display).dpll.lock)) };
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_is_clock_enabled")]
extern "C" fn gen11_dsi_is_clock_enabled(encoder: *mut intel_encoder) -> bool {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: [u32; 5] = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut clock_enabled: bool = false;
let mut phy: i32 = 0;
let tmp: u32 = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
'__slate_break_3: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
if tmp & ((1i32 << unsafe { *((*{
__slate_compound_2 = [10u32, 11u32, 24u32, 4u32, 5u32];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }) as u32) == 0 {
clock_enabled = true;
}
}
phy = (phy as i32).wrapping_add(1) as i32;
}
clock_enabled
}

fn gen11_dsi_map_pll(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: [u32; 5] = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let pll: *mut intel_dpll = unsafe { (*crtc_state).intel_dpll };
let mut phy: i32 = 0;
let mut val: u32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*display).dpll.lock)) };
val = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
let __v2: u32 = val;
val = __v2 & (!ICL_DPCLKA_CFGCR0_DDI_CLK_SEL_MASK!(phy) as u32);
let __v4: u32 = val;
val = __v4 | (ICL_DPCLKA_CFGCR0_DDI_CLK_SEL!(unsafe { (*(*pll).info).id }, phy) as u32);
}
phy = (phy as i32).wrapping_add(1) as i32;
}
intel_de_write(display, unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_9)
} }, val);
phy = 0;
'__slate_break_13: while (phy as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.phys }) as i32) as u64) & BIT__2!(phy) != 0 {
let __v10: u32 = val;
val = __v10 & (!(1i32 << unsafe { *((*{
__slate_compound_12 = [10u32, 11u32, 24u32, 4u32, 5u32];
core::ptr::addr_of_mut!(__slate_compound_12)
}).as_ptr() as *const u32).wrapping_offset((phy as i32) as isize) }) as u32);
}
phy = (phy as i32).wrapping_add(1) as i32;
}
intel_de_write(display, unsafe { *{
__slate_compound_16 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_16)
} }, val);
intel_de_posting_read(display, unsafe { *{
__slate_compound_17 = __SlateRecord_i915_reg_defs_h_14 { reg: 1458816 };
core::ptr::addr_of_mut!(__slate_compound_17)
} });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*display).dpll.lock)) };
}

fn gen11_dsi_configure_transcoder(encoder: *mut intel_encoder, pipe_config: *const intel_crtc_state) {
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_74: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_78: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_82: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_106: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_110: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*pipe_config).uapi.crtc });
let pipe: i32 = unsafe { (*crtc).pipe };
let mut tmp: u32 = 0;
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
'__slate_break_75: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
tmp = intel_de_read(display, unsafe { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438320i32, 440368i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
if unsafe { (*intel_dsi).eot_pkt } {
let __v4: u32 = tmp & 4294967294u32;
tmp = __v4;
} else {
tmp |= 1u32;
}
// enable link calibration if freq > 1.5Gbps
if afe_clk(encoder, pipe_config) >= 1500000i32 {
let __v8: u32 = tmp & 4294967247u32;
tmp = __v8;
tmp |= 32u32;
}
// configure continuous clock
let __v12: u32 = tmp & 4294966527u32;
tmp = __v12;
if unsafe { (*intel_dsi).clock_stop } {
tmp |= 0u32;
} else {
tmp |= 768u32;
}
if (DISPLAY_VER!(display) as i32) >= 12 && unsafe { (*intel_dsi).lp_clock_during_lpm } {
tmp |= 128u32;
} else {
let __v20: u32 = tmp & 4294967167u32;
tmp = __v20;
}
// configure buffer threshold limit to minimum
let __v22: u32 = tmp & 4294964223u32;
tmp = __v22;
tmp |= 0u32;
// set virtual channel to '0'
let __v26: u32 = tmp & 4294955007u32;
tmp = __v26;
let __v27: u32 = tmp;
tmp = __v27 | (PIX_VIRT_CHAN!(0i32) as u32);
// program BGR transmission
if unsafe { (*intel_dsi).bgr_enabled } {
tmp |= 32768u32;
}
// select pixel format
let __v32: u32 = tmp & 4294770687u32;
tmp = __v32;
if unsafe { (*pipe_config).dsc.compression_enable } {
tmp |= 393216u32;
} else {
match (unsafe { (*intel_dsi).pixel_format }) as u32 {
3 => {
tmp |= 0u32;
}
2 => {
tmp |= 65536u32;
}
1 => {
tmp |= 131072u32;
}
0 => {
tmp |= 196608u32;
}
_ => {
let mut __v36: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"intel_dsi->pixel_format".as_ptr() as *mut u8, ((*intel_dsi).pixel_format as u32) as i64) };
let mut __v37: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v37 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v38: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_gen11_dsi_configure_transcoder_39, sym __slate_asm_string_gen11_dsi_configure_transcoder_40, const 757 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v38 = bug;
unsafe { __v37.unwrap()(__v38, c"intel_dsi->pixel_format".as_ptr() as *mut u8, ((*intel_dsi).pixel_format as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v36 = (__ret_warn_on != 0) as i64;
let mut _v115: u32 = tmp;
let mut _v116: u32 = _v115 | 0u32;
tmp = _v116;
}
}
}
let __v50: bool = (DISPLAY_VER!(display) as i32) >= 12 && is_vid_mode(intel_dsi);
if __v50 && unsafe { (*intel_dsi).blanking_pkt } {
tmp |= 4u32;
} else {
let __v54: u32 = tmp & 4294967291u32;
tmp = __v54;
}
// program DSI operation mode
if is_vid_mode(intel_dsi) {
let __v56: u32 = tmp & 3489660927u32;
tmp = __v56;
match unsafe { (*intel_dsi).video_mode } {
2 => {
tmp |= 536870912u32;
}
1 => {
tmp |= 805306368u32;
}
_ => {
let mut __v58: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"intel_dsi->video_mode".as_ptr() as *mut u8, (*intel_dsi).video_mode as i64) };
let mut __v59: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v59 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v60: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t3: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t3, sym __slate_asm_string_gen11_dsi_configure_transcoder_61, sym __slate_asm_string_gen11_dsi_configure_transcoder_62, const 785 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t3 as u64) };
}
__v60 = bug;
unsafe { __v59.unwrap()(__v60, c"intel_dsi->video_mode".as_ptr() as *mut u8, (*intel_dsi).video_mode as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v58 = (__ret_warn_on != 0) as i64;
let _v115: u32 = tmp;
let _v116: u32 = _v115 | 536870912u32;
tmp = _v116;
}
}
} else {
// FIXME: Retrieve this info from VBT.
// As per the spec when dsi transcoder is operating
// in TE GATE mode, TE comes from GPIO
// which is UTIL PIN for DSI 0.
// Also this GPIO would not be used for other
// purposes is an assumption.
let __v69: u32 = tmp & 3489660927u32;
tmp = __v69;
tmp |= 268435456u32;
tmp |= 134217728u32;
}
intel_de_write(display, unsafe { *{
__slate_compound_74 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438320i32, 440368i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_74)
} }, tmp);
}
port = (port as i32).wrapping_add(1) as i32;
}
// enable port sync mode if dual link
if (unsafe { (*intel_dsi).dual_link }) != 0 {
port = 0;
'__slate_break_79: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_rmw(display, unsafe { *{
__slate_compound_78 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(394244) };
core::ptr::addr_of_mut!(__slate_compound_78)
} }, 0u32, REG_BIT__1076!(4i32));
}
port = (port as i32).wrapping_add(1) as i32;
}
// configure stream splitting
configure_dual_link_mode(encoder, pipe_config);
}
port = 0;
'__slate_break_107: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
// select data lane width
tmp = intel_de_read(display, unsafe { *{
__slate_compound_82 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(394240) };
core::ptr::addr_of_mut!(__slate_compound_82)
} });
let __v83: u32 = tmp;
tmp = __v83 & !REG_GENMASK__97!(3i32, 1i32);
let __v85: u32 = tmp;
tmp = __v85 | TRANS_DDI_PORT_WIDTH__2!(unsafe { (*intel_dsi).lane_count });
// select input pipe
let __v88: u32 = tmp & 4294938623u32;
tmp = __v88;
match pipe as i32 {
0 => {
tmp |= 0u32;
}
1 => {
tmp |= 20480u32;
}
2 => {
tmp |= 24576u32;
}
3 => {
tmp |= 28672u32;
}
_ => {
let mut __v90: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"pipe".as_ptr() as *mut u8, (pipe as i32) as i64) };
let mut __v91: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v91 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v92: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t5: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t5, sym __slate_asm_string_gen11_dsi_configure_transcoder_93, sym __slate_asm_string_gen11_dsi_configure_transcoder_94, const 837 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t5 as u64) };
}
__v92 = bug;
unsafe { __v91.unwrap()(__v92, c"pipe".as_ptr() as *mut u8, (pipe as i32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v90 = (__ret_warn_on != 0) as i64;
let _v115: u32 = tmp;
let _v116: u32 = _v115 | 0u32;
tmp = _v116;
}
}
// enable DDI buffer
tmp |= 2147483648u32;
intel_de_write(display, unsafe { *{
__slate_compound_106 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(394240) };
core::ptr::addr_of_mut!(__slate_compound_106)
} }, tmp);
}
port = (port as i32).wrapping_add(1) as i32;
}
// wait for link ready
port = 0;
'__slate_break_112: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
let mut ret: i32 = 0;
dsi_trans = dsi_port_to_transcoder(port);
ret = unsafe { intel_de_wait_for_set_us(display, *{
__slate_compound_110 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438320i32, 440368i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_110)
}, 1048576u32, 2500u32) };
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI link not ready\n".as_ptr() as *const u8) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_710: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_711: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_712: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn gen11_dsi_set_transcoder_timings(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_33: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_37: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let adjusted_mode: *const drm_display_mode = unsafe { core::ptr::addr_of!((*crtc_state).hw.adjusted_mode) };
// program TRANS_HTOTAL register
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
// horizontal timings
let mut htotal: u16 = 0;
let mut hactive: u16 = 0;
let mut hsync_start: u16 = 0;
let mut hsync_end: u16 = 0;
let mut hsync_size: u16 = 0;
let mut hback_porch: u16 = 0;
// vertical timings
let mut vtotal: u16 = 0;
let mut vactive: u16 = 0;
let mut vsync_start: u16 = 0;
let mut vsync_end: u16 = 0;
let mut vsync_shift: u16 = 0;
let mut mul: i32 = 1;
let mut div: i32 = 1;
// Adjust horizontal timings (htotal, hsync_start, hsync_end) to account
// for slower link speed if DSC is enabled.
//
// The compression frequency ratio is the ratio between compressed and
// non-compressed link speeds, and simplifies down to the ratio between
// compressed and non-compressed bpp.
if is_vid_mode(intel_dsi) && unsafe { (*crtc_state).dsc.compression_enable } {
mul = fxp_q4_to_int((unsafe { (*crtc_state).dsc.compressed_bpp_x16 }) as i32);
div = mipi_dsi_pixel_format_to_bpp(unsafe { (*intel_dsi).pixel_format });
}
hactive = unsafe { (*adjusted_mode).crtc_hdisplay };
if is_vid_mode(intel_dsi) {
htotal = (((unsafe { (*adjusted_mode).crtc_htotal }) as i32).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
} else {
htotal = ((hactive as i32).wrapping_add(160).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
}
hsync_start = (((unsafe { (*adjusted_mode).crtc_hsync_start }) as i32).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
hsync_end = (((unsafe { (*adjusted_mode).crtc_hsync_end }) as i32).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
hsync_size = (hsync_end as i32).wrapping_sub(hsync_start as i32) as u16;
hback_porch = ((unsafe { (*adjusted_mode).crtc_htotal }) as i32).wrapping_sub((unsafe { (*adjusted_mode).crtc_hsync_end }) as i32) as u16;
vactive = unsafe { (*adjusted_mode).crtc_vdisplay };
if is_vid_mode(intel_dsi) {
vtotal = unsafe { (*adjusted_mode).crtc_vtotal };
} else {
let mut bpp: i32 = 0;
let mut line_time_us: i32 = 0;
let mut byte_clk_period_ns: i32 = 0;
if unsafe { (*crtc_state).dsc.compression_enable } {
bpp = fxp_q4_to_int((unsafe { (*crtc_state).dsc.compressed_bpp_x16 }) as i32);
} else {
bpp = mipi_dsi_pixel_format_to_bpp(unsafe { (*intel_dsi).pixel_format });
}
byte_clk_period_ns = 1000000 / afe_clk(encoder, crtc_state);
line_time_us = (((htotal as i32).wrapping_mul(bpp / 8).wrapping_mul(byte_clk_period_ns) as u32) / 1000u32.wrapping_mul(unsafe { (*intel_dsi).lane_count })) as i32;
vtotal = (vactive as i32).wrapping_add(400i32.wrapping_add(line_time_us).wrapping_sub(1) / line_time_us) as u16;
}
vsync_start = unsafe { (*adjusted_mode).crtc_vsync_start };
vsync_end = unsafe { (*adjusted_mode).crtc_vsync_end };
vsync_shift = (hsync_start as i32).wrapping_sub((htotal as i32) / 2) as u16;
if (unsafe { (*intel_dsi).dual_link }) != 0 {
hactive = ((hactive as i32) / 2) as u16;
if ((unsafe { (*intel_dsi).dual_link }) as i32) == 1 {
hactive = (hactive as i32).wrapping_add((unsafe { (*intel_dsi).pixel_overlap }) as i32) as u16;
}
htotal = ((htotal as i32) / 2) as u16;
}
// minimum hactive as per bspec: 256 pixels
if ((unsafe { (*adjusted_mode).crtc_hdisplay }) as i32) < 256 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* hactive is less then 256 pixels\n".as_ptr() as *const u8) };
}
// if RGB666 format, then hactive must be multiple of 4 pixels
if ((unsafe { (*intel_dsi).pixel_format }) as u32) == 1 && (hactive as i32) % 4 != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* hactive pixels are not multiple of 4\n".as_ptr() as *const u8) };
}
'__slate_break_10: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_write(display, unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(393216) };
core::ptr::addr_of_mut!(__slate_compound_9)
} }, HACTIVE__2!((hactive as i32).wrapping_sub(1)) | HTOTAL!((htotal as i32).wrapping_sub(1)));
}
port = (port as i32).wrapping_add(1) as i32;
}
// TRANS_HSYNC register to be programmed only for video mode
if is_vid_mode(intel_dsi) {
if (unsafe { (*intel_dsi).video_mode }) == 1 {
// BSPEC: hsync size should be atleast 16 pixels
if (hsync_size as i32) < 16 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* hsync size < 16 pixels\n".as_ptr() as *const u8) };
}
}
if (hback_porch as i32) < 16 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* hback porch < 16 pixels\n".as_ptr() as *const u8) };
}
if (unsafe { (*intel_dsi).dual_link }) != 0 {
hsync_start = ((hsync_start as i32) / 2) as u16;
hsync_end = ((hsync_end as i32) / 2) as u16;
}
port = 0;
'__slate_break_20: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_write(display, unsafe { *{
__slate_compound_19 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(393224) };
core::ptr::addr_of_mut!(__slate_compound_19)
} }, HSYNC_START__3!((hsync_start as i32).wrapping_sub(1)) | HSYNC_END__2!((hsync_end as i32).wrapping_sub(1)));
}
port = (port as i32).wrapping_add(1) as i32;
}
}
// program TRANS_VTOTAL register
port = 0;
'__slate_break_24: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
// FIXME: Programming this by assuming progressive mode, since
// non-interlaced info from VBT is not saved inside
// struct drm_display_mode.
// For interlace mode: program required pixel minus 2
intel_de_write(display, unsafe { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(393228) };
core::ptr::addr_of_mut!(__slate_compound_23)
} }, VACTIVE__2!((vactive as i32).wrapping_sub(1)) | VTOTAL!((vtotal as i32).wrapping_sub(1)));
}
port = (port as i32).wrapping_add(1) as i32;
}
if (vsync_end as i32) < (vsync_start as i32) || (vsync_end as i32) > (vtotal as i32) {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* Invalid vsync_end value\n".as_ptr() as *const u8) };
}
if (vsync_start as i32) < (vactive as i32) {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* vsync_start less than vactive\n".as_ptr() as *const u8) };
}
// program TRANS_VSYNC register for video mode only
if is_vid_mode(intel_dsi) {
port = 0;
'__slate_break_30: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_write(display, unsafe { *{
__slate_compound_29 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(393236) };
core::ptr::addr_of_mut!(__slate_compound_29)
} }, VSYNC_START!((vsync_start as i32).wrapping_sub(1)) | VSYNC_END__3!((vsync_end as i32).wrapping_sub(1)));
}
port = (port as i32).wrapping_add(1) as i32;
}
}
// FIXME: It has to be programmed only for video modes and interlaced
// modes. Put the check condition here once interlaced
// info available as described above.
// program TRANS_VSYNCSHIFT register
if is_vid_mode(intel_dsi) {
port = 0;
'__slate_break_34: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_write(display, unsafe { *{
__slate_compound_33 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(393256) };
core::ptr::addr_of_mut!(__slate_compound_33)
} }, vsync_shift as u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
}
// program TRANS_VBLANK register, should be same as vtotal programmed
//
// FIXME get rid of these local hacks and do it right,
// this will not handle eg. delayed vblank correctly.
if (DISPLAY_VER!(display) as i32) >= 12 {
port = 0;
'__slate_break_38: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_write(display, unsafe { *{
__slate_compound_37 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(393232) };
core::ptr::addr_of_mut!(__slate_compound_37)
} }, VBLANK_START__2!((vactive as i32).wrapping_sub(1)) | VBLANK_END__2!((vtotal as i32).wrapping_sub(1)));
}
port = (port as i32).wrapping_add(1) as i32;
}
}
}

fn gen11_dsi_enable_transcoder(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
'__slate_break_4: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_PIPE_OFFSET!(display, dsi_trans).wrapping_add(458760) };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 0u32, REG_BIT__462!(31i32));
// wait for transcoder to be enabled
if (unsafe { intel_de_wait_for_set_ms(display, *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_PIPE_OFFSET!(display, dsi_trans).wrapping_add(458760) };
core::ptr::addr_of_mut!(__slate_compound_2)
}, REG_BIT__1032!(30i32), 10u32) }) != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI transcoder not enabled\n".as_ptr() as *const u8) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_setup_timeouts(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
let mut hs_tx_timeout: u32 = 0;
let mut lp_rx_timeout: u32 = 0;
let mut ta_timeout: u32 = 0;
let mut divisor: u32 = 0;
let mut mul: u32 = 0;
// escape clock count calculation:
// BYTE_CLK_COUNT = TIME_NS/(8 * UI)
// UI (nsec) = (10^6)/Bitrate
// TIME_NS = (BYTE_CLK_COUNT * 8 * 10^6)/ Bitrate
// ESCAPE_CLK_COUNT  = TIME_NS/ESC_CLK_NS
divisor = unsafe { intel_dsi_tlpx_ns(intel_dsi as *const intel_dsi) }.wrapping_mul(afe_clk(encoder, crtc_state)).wrapping_mul(1000) as u32;
mul = 8000000u32;
hs_tx_timeout = (((unsafe { (*intel_dsi).hs_tx_timeout }) as i32) as u32).wrapping_mul(mul).wrapping_add(divisor).wrapping_sub(1) / divisor;
lp_rx_timeout = (((unsafe { (*intel_dsi).lp_rx_timeout }) as i32) as u32).wrapping_mul(mul).wrapping_add(divisor).wrapping_sub(1) / divisor;
ta_timeout = (((unsafe { (*intel_dsi).turn_arnd_val }) as i32) as u32).wrapping_mul(mul).wrapping_add(divisor).wrapping_sub(1) / divisor;
'__slate_break_4: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
// program hst_tx_timeout
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438340i32, 440388i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 4294901760u32, HSTX_TIMEOUT_VALUE!(hs_tx_timeout));
// FIXME: DSI_CALIB_TO
// program lp_rx_host timeout
intel_de_rmw(display, unsafe { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438344i32, 440392i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} }, 65535u32, LPRX_TIMEOUT_VALUE!(lp_rx_timeout));
// FIXME: DSI_PWAIT_TO
// program turn around timeout
intel_de_rmw(display, unsafe { *{
__slate_compound_3 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438348i32, 440396i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_3)
} }, 65535u32, TA_TIMEOUT_VALUE!(ta_timeout));
}
port = (port as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_config_util_pin(encoder: *mut intel_encoder, enable: bool) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut tmp: u32 = 0;
// used as TE i/p for DSI0,
// for dual link/DSI1 TE is from slave DSI1
// through GPIO.
if is_vid_mode(intel_dsi) || (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT!(1i32) != 0 {
return;
}
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: 295936 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
if enable {
tmp |= 524288u32;
tmp |= 2147483648u32;
} else {
let __v7: u32 = tmp & 2147483647u32;
tmp = __v7;
}
intel_de_write(display, unsafe { *{
__slate_compound_8 = __SlateRecord_i915_reg_defs_h_14 { reg: 295936 };
core::ptr::addr_of_mut!(__slate_compound_8)
} }, tmp);
}

fn gen11_dsi_enable_port_and_phy(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
// step 4a: power up all lanes of the DDI used by DSI
gen11_dsi_power_up_lanes(encoder);
// step 4b: configure lane sequencing of the Combo-PHY transmitters
gen11_dsi_config_phy_lanes_sequence(encoder);
// step 4c: configure voltage swing and skew
gen11_dsi_voltage_swing_program_seq(encoder);
// setup D-PHY timings
gen11_dsi_setup_dphy_timings(encoder, crtc_state);
// enable DDI buffer
gen11_dsi_enable_ddi_buffer(encoder);
gen11_dsi_gate_clocks(encoder);
gen11_dsi_setup_timings(encoder, crtc_state);
// Since transcoder is configured to take events from GPIO
gen11_dsi_config_util_pin(encoder, true);
// step 4h: setup DSI protocol timeouts
gen11_dsi_setup_timeouts(encoder, crtc_state);
// Step (4h, 4i, 4j, 4k): Configure transcoder
gen11_dsi_configure_transcoder(encoder, crtc_state);
}

fn gen11_dsi_powerup_panel(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
// set maximum return packet size
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
let mut tmp: u32 = 0;
let mut ret: i32 = 0;
'__slate_break_5: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
// FIXME: This uses the number of DW's currently in the payload
// receive queue. This is probably not what we want here.
tmp = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438484i32, 440532i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
tmp &= 255u32;
// multiply "Number Rx Payload DW" by 4 to get max value
tmp = tmp.wrapping_mul(4);
let dsi: *mut mipi_dsi_device = unsafe { (*(*((*intel_dsi).dsi_hosts.as_mut_ptr() as *mut *mut intel_dsi_host).wrapping_offset((port as i32) as isize))).device };
ret = unsafe { mipi_dsi_set_maximum_return_packet_size(dsi, tmp as u16) };
if ret < 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* error setting max return pkt size%d\n".as_ptr() as *const u8, tmp) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 2u32) };
// ensure all panel commands dispatched before enabling transcoder
wait_for_cmds_dispatched_to_panel(encoder);
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_pre_pll_enable")]
extern "C" fn gen11_dsi_pre_pll_enable(state: *mut intel_atomic_state, encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
unsafe { intel_dsi_wait_panel_power_cycle(intel_dsi) };
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 10u32) };
unsafe { msleep((*intel_dsi).panel_on_delay as u32) };
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 1u32) };
// step2: enable IO power
gen11_dsi_enable_io_power(encoder);
// step3: enable DSI PLL
gen11_dsi_program_esc_clk_div(encoder, crtc_state);
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_pre_enable")]
extern "C" fn gen11_dsi_pre_enable(state: *mut intel_atomic_state, encoder: *mut intel_encoder, pipe_config: *const intel_crtc_state, conn_state: *const drm_connector_state) {
// step3b
gen11_dsi_map_pll(encoder, pipe_config);
// step4: enable DSI port and DPHY
gen11_dsi_enable_port_and_phy(encoder, pipe_config);
// step5: program and powerup panel
gen11_dsi_powerup_panel(encoder);
unsafe { intel_dsc_dsi_pps_write(encoder, pipe_config) };
// step6c: configure transcoder timings
gen11_dsi_set_transcoder_timings(encoder, pipe_config);
}

/// Wa_1409054076:icl,jsl,ehl
/// When pipe A is disabled and MIPI DSI is enabled on pipe B,
/// the AMT KVMR feature will incorrectly see pipe A as enabled.
/// Set 0x42080 bit 23=1 before enabling DSI on pipe B and leave
/// it set while DSI is enabled on pipe B
fn icl_apply_kvmr_pipe_a_wa(encoder: *mut intel_encoder, pipe: i32, enable: bool) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
if (DISPLAY_VER!(display) as i32) == 11 && (pipe as i32) == 1 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: 270464 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_BIT__1200!(23i32), if enable { REG_BIT__1317!(23i32) } else { 0u32 });
}
}

/// Wa_16012360555:adl-p
/// SW will have to program the "LP to HS Wakeup Guardband"
/// to account for the repeaters on the HS Request/Ready
/// PPI signaling between the Display engine and the DPHY.
fn adlp_set_lp_hs_wakeup_gb(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
if (DISPLAY_VER!(display) as i32) == 13 {
port = 0;
'__slate_break_2: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438464i32, 440512i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_GENMASK__67!(15i32, 12i32), TGL_DSI_CHKN_LSHS_GB!(4i32));
}
port = (port as i32).wrapping_add(1) as i32;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_enable")]
extern "C" fn gen11_dsi_enable(state: *mut intel_atomic_state, encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state, conn_state: *const drm_connector_state) {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*crtc_state).uapi.crtc });
// Wa_1409054076:icl,jsl,ehl
icl_apply_kvmr_pipe_a_wa(encoder, unsafe { (*crtc).pipe }, true);
// Wa_16012360555:adl-p
adlp_set_lp_hs_wakeup_gb(encoder);
// step6d: enable dsi transcoder
gen11_dsi_enable_transcoder(encoder);
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 3u32) };
// step7: enable backlight
unsafe { intel_backlight_enable(crtc_state, conn_state) };
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 6u32) };
unsafe { intel_panel_prepare(crtc_state, conn_state) };
unsafe { intel_crtc_vblank_on(crtc_state) };
}

fn gen11_dsi_disable_transcoder(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
'__slate_break_4: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
// disable transcoder
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_PIPE_OFFSET!(display, dsi_trans).wrapping_add(458760) };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_BIT__455!(31i32), 0u32);
// wait for transcoder to be disabled
if (unsafe { intel_de_wait_for_clear_ms(display, *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_PIPE_OFFSET!(display, dsi_trans).wrapping_add(458760) };
core::ptr::addr_of_mut!(__slate_compound_2)
}, REG_BIT__780!(30i32), 50u32) }) != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI trancoder not disabled\n".as_ptr() as *const u8) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
}

fn gen11_dsi_powerdown_panel(encoder: *mut intel_encoder) {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 4u32) };
// ensure cmds dispatched to panel
wait_for_cmds_dispatched_to_panel(encoder);
}

fn gen11_dsi_deconfigure_trancoder(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_20: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let mut dsi_trans: i32 = 0;
let mut tmp: u32 = 0;
let mut ret: i32 = 0;
// disable periodic update mode
if is_cmd_mode(intel_dsi) {
port = 0;
'__slate_break_2: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438324i32, 440372i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, 536870912u32, 0u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
}
// put dsi link in ULPS
port = 0;
'__slate_break_13: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
tmp = intel_de_read(display, unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438488i32, 440536i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
tmp |= 1u32;
let __v9: u32 = tmp & 4294967039u32;
tmp = __v9;
intel_de_write(display, unsafe { *{
__slate_compound_10 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438488i32, 440536i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_10)
} }, tmp);
ret = unsafe { intel_de_wait_for_set_us(display, *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438488i32, 440536i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_11)
}, 65536u32, 10u32) };
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI link not in ULPS\n".as_ptr() as *const u8) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
// disable ddi function
port = 0;
'__slate_break_17: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_rmw(display, unsafe { *{
__slate_compound_16 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(394240) };
core::ptr::addr_of_mut!(__slate_compound_16)
} }, 2147483648u32, 0u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
// disable port sync mode if dual link
if (unsafe { (*intel_dsi).dual_link }) != 0 {
port = 0;
'__slate_break_21: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
dsi_trans = dsi_port_to_transcoder(port);
intel_de_rmw(display, unsafe { *{
__slate_compound_20 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, dsi_trans).wrapping_add(394244) };
core::ptr::addr_of_mut!(__slate_compound_20)
} }, REG_BIT__757!(4i32), 0u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
}
}

fn gen11_dsi_disable_port(encoder: *mut intel_encoder) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
let mut ret: i32 = 0;
gen11_dsi_ungate_clocks(encoder);
'__slate_break_4: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 409600i32, 409856i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_BIT__1441!(31i32), 0u32);
ret = unsafe { intel_de_wait_for_set_us(display, *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 409600i32, 409856i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
}, REG_BIT__127!(7i32), 8u32) };
if ret != 0 {
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DDI port:%c buffer not idle\n".as_ptr() as *const u8, port_name!(port)) };
}
}
port = (port as i32).wrapping_add(1) as i32;
}
gen11_dsi_gate_clocks(encoder);
}

fn gen11_dsi_disable_io_power(encoder: *mut intel_encoder) {
let mut __slate_compound_6: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let mut port: i32 = 0;
while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
let wakeref: *mut ref_tracker = fetch_and_zero!(unsafe { (*intel_dsi).io_wakeref.as_mut_ptr() as *mut *mut ref_tracker }.wrapping_offset((port as i32) as isize));
intel_display_power_put(display, (if (port as i32) == 0 { 29i32 } else { 30i32 }) as u32, wakeref);
}
port = (port as i32).wrapping_add(1) as i32;
}
// set mode to DDI
port = 0;
'__slate_break_7: while (port as i32) < 9 {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) & BIT__2!(port) != 0 {
intel_de_rmw(display, unsafe { *{
__slate_compound_6 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(port, 438420i32, 440468i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_6)
} }, 1u32, 0u32);
}
port = (port as i32).wrapping_add(1) as i32;
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_disable")]
extern "C" fn gen11_dsi_disable(state: *mut intel_atomic_state, encoder: *mut intel_encoder, old_crtc_state: *const intel_crtc_state, old_conn_state: *const drm_connector_state) {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
unsafe { intel_panel_unprepare(old_conn_state) };
// step1: turn off backlight
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 7u32) };
unsafe { intel_backlight_disable(old_conn_state) };
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_post_disable")]
extern "C" fn gen11_dsi_post_disable(state: *mut intel_atomic_state, encoder: *mut intel_encoder, old_crtc_state: *const intel_crtc_state, old_conn_state: *const drm_connector_state) {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*old_crtc_state).uapi.crtc });
unsafe { intel_crtc_vblank_off(old_crtc_state) };
// step2d,e: disable transcoder and wait
gen11_dsi_disable_transcoder(encoder);
// Wa_1409054076:icl,jsl,ehl
icl_apply_kvmr_pipe_a_wa(encoder, unsafe { (*crtc).pipe }, false);
// step2f,g: powerdown panel
gen11_dsi_powerdown_panel(encoder);
// step2h,i,j: deconfig trancoder
gen11_dsi_deconfigure_trancoder(encoder);
unsafe { intel_dsc_disable(old_crtc_state) };
unsafe { skl_scaler_disable(old_crtc_state) };
// step3: disable port
gen11_dsi_disable_port(encoder);
gen11_dsi_config_util_pin(encoder, false);
// step4: disable IO power
gen11_dsi_disable_io_power(encoder);
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 5u32) };
unsafe { msleep((*intel_dsi).panel_off_delay as u32) };
unsafe { intel_dsi_vbt_exec_sequence(intel_dsi, 11u32) };
unsafe {
(*intel_dsi).panel_power_off_time = ktime_get_boottime();
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_mode_valid")]
extern "C" fn gen11_dsi_mode_valid(connector: *mut drm_connector, mode: *const drm_display_mode) -> i32 {
let display: *mut intel_display = to_intel_display__4!(unsafe { (*connector).dev });
let status: i32 = unsafe { intel_cpu_transcoder_mode_valid(display, mode) };
if (status as i32) != 0 {
return status;
}
// FIXME: DSC?
unsafe { intel_dsi_mode_valid(connector, mode) }
}

fn gen11_dsi_get_timings(encoder: *mut intel_encoder, pipe_config: *mut intel_crtc_state) {
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let adjusted_mode: *mut drm_display_mode = unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) };
if is_vid_mode(intel_dsi) && (unsafe { (*pipe_config).dsc.compressed_bpp_x16 }) != 0 {
let div: i32 = fxp_q4_to_int((unsafe { (*pipe_config).dsc.compressed_bpp_x16 }) as i32);
let mul: i32 = mipi_dsi_pixel_format_to_bpp(unsafe { (*intel_dsi).pixel_format });
unsafe {
(*adjusted_mode).crtc_htotal = (((*adjusted_mode).crtc_htotal as i32).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
}
unsafe {
(*adjusted_mode).crtc_hsync_start = (((*adjusted_mode).crtc_hsync_start as i32).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
}
unsafe {
(*adjusted_mode).crtc_hsync_end = (((*adjusted_mode).crtc_hsync_end as i32).wrapping_mul(mul).wrapping_add(div).wrapping_sub(1) / div) as u16;
}
}
if (unsafe { (*intel_dsi).dual_link }) != 0 {
unsafe {
(*adjusted_mode).crtc_hdisplay = ((*adjusted_mode).crtc_hdisplay as i32).wrapping_mul(2) as u16;
}
if ((unsafe { (*intel_dsi).dual_link }) as i32) == 1 {
unsafe {
(*adjusted_mode).crtc_hdisplay = ((*adjusted_mode).crtc_hdisplay as i32).wrapping_sub((*intel_dsi).pixel_overlap as i32) as u16;
}
}
unsafe {
(*adjusted_mode).crtc_htotal = ((*adjusted_mode).crtc_htotal as i32).wrapping_mul(2) as u16;
}
}
unsafe {
(*adjusted_mode).crtc_hblank_start = (*adjusted_mode).crtc_hdisplay;
}
unsafe {
(*adjusted_mode).crtc_hblank_end = (*adjusted_mode).crtc_htotal;
}
if ((unsafe { (*intel_dsi).operation_mode }) as i32) == 0 {
if (unsafe { (*intel_dsi).dual_link }) != 0 {
unsafe {
(*adjusted_mode).crtc_hsync_start = ((*adjusted_mode).crtc_hsync_start as i32).wrapping_mul(2) as u16;
}
unsafe {
(*adjusted_mode).crtc_hsync_end = ((*adjusted_mode).crtc_hsync_end as i32).wrapping_mul(2) as u16;
}
}
}
unsafe {
(*adjusted_mode).crtc_vblank_start = (*adjusted_mode).crtc_vdisplay;
}
unsafe {
(*adjusted_mode).crtc_vblank_end = (*adjusted_mode).crtc_vtotal;
}
}

fn gen11_dsi_is_periodic_cmd_mode(intel_dsi: *mut intel_dsi) -> bool {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(unsafe { core::ptr::addr_of_mut!((*intel_dsi).base) });
let mut dsi_trans: i32 = 0;
let mut val: u32 = 0;
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) == BIT!(1i32) {
dsi_trans = 6;
} else {
dsi_trans = 5;
}
val = intel_de_read(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _TRANS__2!((dsi_trans as i32).wrapping_sub(5), 438320i32, 440368i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} });
val & 536870912u32 != 0
}

fn gen11_dsi_get_cmd_mode_config(intel_dsi: *mut intel_dsi, pipe_config: *mut intel_crtc_state) {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) == BIT!(1i32) | BIT!(0i32) {
unsafe {
(*pipe_config).mode_flags = (((*pipe_config).mode_flags as i32) | 24i32) as u8;
}
} else {
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) == BIT!(1i32) {
unsafe {
(*pipe_config).mode_flags = (((*pipe_config).mode_flags as i32) | 16i32) as u8;
}
} else {
unsafe {
(*pipe_config).mode_flags = (((*pipe_config).mode_flags as i32) | 8i32) as u8;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_get_config")]
extern "C" fn gen11_dsi_get_config(encoder: *mut intel_encoder, pipe_config: *mut intel_crtc_state) {
let crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*pipe_config).uapi.crtc });
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
unsafe { intel_ddi_get_clock(encoder, pipe_config, icl_ddi_combo_get_pll(encoder)) };
unsafe {
(*pipe_config).hw.adjusted_mode.crtc_clock = (*intel_dsi).pclk as i32;
}
if (unsafe { (*intel_dsi).dual_link }) != 0 {
let __v2: i32 = unsafe { (*pipe_config).hw.adjusted_mode.crtc_clock };
unsafe {
(*pipe_config).hw.adjusted_mode.crtc_clock = __v2.wrapping_mul(2);
}
}
gen11_dsi_get_timings(encoder, pipe_config);
let __v5: u32 = unsafe { (*pipe_config).output_types };
let __v6: u32 = ((__v5 as u64) | BIT!(9i32)) as u32;
unsafe {
(*pipe_config).output_types = __v6;
}
unsafe {
(*pipe_config).pipe_bpp = bdw_get_pipe_misc_bpp(crtc);
}
// Get the details on which TE should be enabled
if is_cmd_mode(intel_dsi) {
gen11_dsi_get_cmd_mode_config(intel_dsi, pipe_config);
}
if gen11_dsi_is_periodic_cmd_mode(intel_dsi) {
unsafe {
(*pipe_config).mode_flags = (((*pipe_config).mode_flags as i32) | 32i32) as u8;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_sync_state")]
extern "C" fn gen11_dsi_sync_state(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let mut __slate_compound_3: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let mut pipe: i32 = 0;
if crtc_state.is_null() {
return;
}
let intel_crtc: *mut intel_crtc = to_intel_crtc!(unsafe { (*crtc_state).uapi.crtc });
pipe = unsafe { (*intel_crtc).pipe };
// wa verify 1409054076:icl,jsl,ehl
let __v2: bool = (DISPLAY_VER!(display) as i32) == 11 && (pipe as i32) == 1 && intel_de_read(display, unsafe { *{
__slate_compound_3 = __SlateRecord_i915_reg_defs_h_14 { reg: 270464 };
core::ptr::addr_of_mut!(__slate_compound_3)
} }) & REG_BIT__682!(23i32) == 0;
if __v2 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"[ENCODER:%d:%s] BIOS left IGNORE_KVMR_PIPE_A cleared with pipe B enabled\n".as_ptr() as *const u8, (*encoder).base.base.id, (*encoder).base.name) };
}
}

fn gen11_dsi_dsc_compute_config(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state) -> i32 {
let display: *mut intel_display = to_intel_display!(encoder);
let vdsc_cfg: *mut drm_dsc_config = unsafe { core::ptr::addr_of_mut!((*crtc_state).dsc.config) };
let dsc_max_bpc: i32 = if (DISPLAY_VER!(display) as i32) >= 12 { 12 } else { 10 };
let mut use_dsc: bool = false;
let mut ret: i32 = 0;
use_dsc = unsafe { intel_bios_get_dsc_params(encoder, crtc_state, dsc_max_bpc) };
if !use_dsc {
return 0;
}
if (unsafe { (*crtc_state).pipe_bpp }) < 24i32 {
return 22i32.wrapping_neg();
}
// FIXME: initialize from VBT
unsafe {
(*vdsc_cfg).rc_model_size = 8192;
}
unsafe {
(*vdsc_cfg).pic_height = (*crtc_state).hw.adjusted_mode.crtc_vdisplay;
}
ret = unsafe { intel_dsc_compute_params(crtc_state) };
if ret != 0 {
return ret;
}
// DSI specific sanity checks on the common code
let mut __v1: i64;
{
let __ret_warn_on: i32 = !(!unsafe { (*vdsc_cfg).vbr_enable }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->vbr_enable)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_gen11_dsi_dsc_compute_config_4, sym __slate_asm_string_gen11_dsi_dsc_compute_config_5, const 1643 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->vbr_enable)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
let mut __v7: i64;
{
let __ret_warn_on: i32 = !(!unsafe { (*vdsc_cfg).simple_422 }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->simple_422)".as_ptr() as *mut u8) };
let mut __v8: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v8 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v9: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_gen11_dsi_dsc_compute_config_10, sym __slate_asm_string_gen11_dsi_dsc_compute_config_11, const 1644 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
}
__v9 = bug;
unsafe { __v8.unwrap()(__v9, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->simple_422)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
}
let mut __v13: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*vdsc_cfg).pic_width }) as i32) % ((unsafe { (*vdsc_cfg).slice_width }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->pic_width % vdsc_cfg->slice_width)".as_ptr() as *mut u8) };
let mut __v14: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v14 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v15: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t2: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t2, sym __slate_asm_string_gen11_dsi_dsc_compute_config_16, sym __slate_asm_string_gen11_dsi_dsc_compute_config_17, const 1646 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t2 as u64) };
}
__v15 = bug;
unsafe { __v14.unwrap()(__v15, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->pic_width % vdsc_cfg->slice_width)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v13 = (__ret_warn_on != 0) as i64;
}
let mut __v19: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*vdsc_cfg).slice_height }) as i32) < 8) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->slice_height < 8)".as_ptr() as *mut u8) };
let mut __v20: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v20 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v21: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t3: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t3, sym __slate_asm_string_gen11_dsi_dsc_compute_config_22, sym __slate_asm_string_gen11_dsi_dsc_compute_config_23, const 1647 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t3 as u64) };
}
__v21 = bug;
unsafe { __v20.unwrap()(__v21, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->slice_height < 8)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v19 = (__ret_warn_on != 0) as i64;
}
let mut __v25: i64;
let __ret_warn_on: i32 = (((unsafe { (*vdsc_cfg).pic_height }) as i32) % ((unsafe { (*vdsc_cfg).slice_height }) as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->pic_height % vdsc_cfg->slice_height)".as_ptr() as *mut u8) };
let mut __v26: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v26 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v27: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t4: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t4, sym __slate_asm_string_gen11_dsi_dsc_compute_config_28, sym __slate_asm_string_gen11_dsi_dsc_compute_config_29, const 1649 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t4 as u64) };
}
__v27 = bug;
unsafe { __v26.unwrap()(__v27, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(vdsc_cfg->pic_height % vdsc_cfg->slice_height)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v25 = (__ret_warn_on != 0) as i64;
ret = unsafe { drm_dsc_compute_rc_parameters(vdsc_cfg) };
if ret != 0 {
return ret;
}
unsafe { intel_dsc_enable_on_crtc(crtc_state) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_713: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_714: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_715: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_716: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_717: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_compute_config")]
extern "C" fn gen11_dsi_compute_config(state: *mut intel_atomic_state, encoder: *mut intel_encoder, pipe_config: *mut intel_crtc_state, conn_state: *mut drm_connector_state) -> i32 {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dsi: *mut intel_dsi = enc_to_intel_dsi(encoder);
let intel_connector: *mut intel_connector = unsafe { (*intel_dsi).attached_connector };
let adjusted_mode: *mut drm_display_mode = unsafe { core::ptr::addr_of_mut!((*pipe_config).hw.adjusted_mode) };
let mut ret: i32 = 0;
unsafe {
(*pipe_config).sink_format = 0;
}
unsafe {
(*pipe_config).output_format = 0;
}
ret = unsafe { intel_panel_compute_config(state, pipe_config, intel_connector) };
if ret != 0 {
return ret;
}
ret = unsafe { intel_pfit_compute_config(pipe_config, conn_state as *const drm_connector_state) };
if ret != 0 {
return ret;
}
unsafe {
(*adjusted_mode).flags = 0;
}
// Dual link goes to trancoder DSI'0'
if (((unsafe { (*intel_dsi).__slate_anon_6.ports }) as i32) as u64) == BIT!(1i32) {
unsafe {
(*pipe_config).cpu_transcoder = 6;
}
} else {
unsafe {
(*pipe_config).cpu_transcoder = 5;
}
}
if ((unsafe { (*intel_dsi).pixel_format }) as u32) == 0 {
unsafe {
(*pipe_config).pipe_bpp = 24;
}
} else {
unsafe {
(*pipe_config).pipe_bpp = 18;
}
}
unsafe {
(*pipe_config).clock_set = true;
}
if gen11_dsi_dsc_compute_config(encoder, pipe_config) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"Attempting to use DSC failed\n".as_ptr() as *const u8) };
}
unsafe {
(*pipe_config).port_clock = afe_clk(encoder, pipe_config as *const intel_crtc_state) / 5;
}
// In case of TE GATE cmd mode, we
// receive TE from the slave if
// dual link is enabled
if is_cmd_mode(intel_dsi) {
gen11_dsi_get_cmd_mode_config(intel_dsi, pipe_config);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_get_power_domains")]
extern "C" fn gen11_dsi_get_power_domains(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state) {
get_dsi_io_power_domains(enc_to_intel_dsi(encoder));
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_get_hw_state")]
extern "C" fn gen11_dsi_get_hw_state(encoder: *mut intel_encoder, pipe: *mut i32) -> bool {
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut tmp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(tmp) as *mut u32;
let mut ret: bool = unsafe { core::mem::zeroed() };
let mut port: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(port) as *mut i32;
let mut wakeref: core::mem::MaybeUninit<*mut ref_tracker> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ref_tracker = core::ptr::addr_of_mut!(wakeref) as *mut *mut ref_tracker;
let mut dsi_trans: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(dsi_trans) as *mut i32;
let mut intel_dsi: *mut intel_dsi = unsafe { core::mem::zeroed() };
let mut display: *mut intel_display = unsafe { core::mem::zeroed() };
unsafe {
display = to_intel_display!(encoder);
intel_dsi = enc_to_intel_dsi(encoder);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
ret = false;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = intel_display_power_get_if_enabled(display, (*encoder).power_domain);
if (*__slate_slot_5).is_null() {
return false;
} else {
*__slate_slot_4 = 0;
'__join_0: {
'__loop_1: loop {
if (*__slate_slot_4 as i32) < 9 {
if (((*intel_dsi).__slate_anon_6.ports as i32) as u64) & BIT__2!(*__slate_slot_4) != 0 {
*__slate_slot_6 = dsi_port_to_transcoder(*__slate_slot_4);
*__slate_slot_2 = intel_de_read(display, *{
__slate_compound_10 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_TRANS_OFFSET!(display, *__slate_slot_6).wrapping_add(394240) };
core::ptr::addr_of_mut!(__slate_compound_10)
});
let __t0: u32 = *__slate_slot_2 & 28672u32;
if __t0 == 0 {
*pipe = 0;
} else {
if __t0 == 20480 {
*pipe = 1;
} else {
if __t0 == 24576 {
*pipe = 2;
} else {
if __t0 == 28672 {
*pipe = 3;
} else {
break '__loop_1;
}
}
}
}
*__slate_slot_2 = intel_de_read(display, *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: INTEL_DISPLAY_DEVICE_PIPE_OFFSET!(display, *__slate_slot_6).wrapping_add(458760) };
core::ptr::addr_of_mut!(__slate_compound_9)
});
ret = *__slate_slot_2 & REG_BIT__1449!(31i32) != 0;
}
*__slate_slot_4 = (*__slate_slot_4 as i32).wrapping_add(1) as i32;
} else {
break '__join_0;
}
}
while false {
}
_dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* Invalid PIPE input\n".as_ptr() as *const u8);
}
intel_display_power_put(display, (*encoder).power_domain, *__slate_slot_5);
return ret;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_initial_fastset_check")]
extern "C" fn gen11_dsi_initial_fastset_check(encoder: *mut intel_encoder, crtc_state: *mut intel_crtc_state) -> bool {
if unsafe { (*crtc_state).dsc.compression_enable } {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*encoder).base.dev as *const drm_device) as *const device, 2u32, c"Forcing full modeset due to DSC being enabled\n".as_ptr() as *const u8) };
unsafe {
let __t0 = true;
(*crtc_state).uapi.__slate_bits_0.__set_mode_changed(__t0);
}
return false;
}
true
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_encoder_destroy")]
extern "C" fn gen11_dsi_encoder_destroy(encoder: *mut drm_encoder) {
unsafe { intel_encoder_destroy(encoder) };
}

static gen11_dsi_encoder_funcs: __SlateConst<drm_encoder_funcs> = __SlateConst({
let mut __t0: drm_encoder_funcs = unsafe { core::mem::zeroed() };
__t0.destroy = Some(gen11_dsi_encoder_destroy);
__t0
});

static gen11_dsi_connector_funcs: __SlateConst<drm_connector_funcs> = __SlateConst({
let mut __t0: drm_connector_funcs = unsafe { core::mem::zeroed() };
__t0.detect = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, bool) -> u32>>(intel_panel_detect as *const ()) };
__t0.fill_modes = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>>(drm_helper_probe_single_connector_modes as *const ()) };
__t0.late_register = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector) -> i32>>(intel_connector_register as *const ()) };
__t0.early_unregister = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector)>>(intel_connector_unregister as *const ()) };
__t0.destroy = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector)>>(intel_connector_destroy as *const ()) };
__t0.atomic_duplicate_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>>(intel_digital_connector_duplicate_state as *const ()) };
__t0.atomic_destroy_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state)>>(drm_atomic_helper_connector_destroy_state as *const ()) };
__t0.atomic_set_property = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state, *mut drm_property, u64) -> i32>>(intel_digital_connector_atomic_set_property as *const ()) };
__t0.atomic_get_property = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *const drm_connector_state, *mut drm_property, *mut u64) -> i32>>(intel_digital_connector_atomic_get_property as *const ()) };
__t0
});

static gen11_dsi_connector_helper_funcs: __SlateConst<drm_connector_helper_funcs> = __SlateConst({
let mut __t0: drm_connector_helper_funcs = unsafe { core::mem::zeroed() };
__t0.get_modes = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector) -> i32>>(intel_dsi_get_modes as *const ()) };
__t0.mode_valid = Some(gen11_dsi_mode_valid);
__t0.atomic_check = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_atomic_commit) -> i32>>(intel_digital_connector_atomic_check as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_host_attach")]
extern "C" fn gen11_dsi_host_attach(host: *mut mipi_dsi_host, dsi: *mut mipi_dsi_device) -> i32 {
0
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_host_detach")]
extern "C" fn gen11_dsi_host_detach(host: *mut mipi_dsi_host, dsi: *mut mipi_dsi_device) -> i32 {
0
}

#[unsafe(link_section = ".text.slate_distinct.icl_dsi.gen11_dsi_host_transfer")]
extern "C" fn gen11_dsi_host_transfer(host: *mut mipi_dsi_host, msg: *const mipi_dsi_msg) -> i64 {
let intel_dsi_host: *mut intel_dsi_host = to_intel_dsi_host(host);
let mut dsi_pkt: mipi_dsi_packet = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(dsi_pkt) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut ret: i64 = (unsafe { mipi_dsi_create_packet(core::ptr::addr_of_mut!(dsi_pkt), msg) }) as i64;
let mut enable_lpdt: bool = false;
if ret < 0 {
return ret;
}
if (((unsafe { (*msg).flags }) as i32) as u64) & BIT!(1i32) != 0 {
enable_lpdt = true;
}
// only long packet contains payload
if unsafe { mipi_dsi_packet_format_is_long((*msg).r#type) } {
ret = dsi_send_pkt_payld(intel_dsi_host, unsafe { &(*(core::ptr::addr_of_mut!(dsi_pkt) as *const mipi_dsi_packet)) }) as i64;
if ret < 0 {
return ret;
}
}
// send packet header
ret = dsi_send_pkt_hdr(intel_dsi_host, unsafe { &(*(core::ptr::addr_of_mut!(dsi_pkt) as *const mipi_dsi_packet)) }, enable_lpdt) as i64;
if ret < 0 {
return ret;
}
// TODO: add payload receive code if needed
ret = (core::mem::size_of::<[u8; 4]>() as u64).wrapping_add(dsi_pkt.payload_length) as i64;
ret
}

static gen11_dsi_host_ops: __SlateConst<mipi_dsi_host_ops> = __SlateConst(mipi_dsi_host_ops { attach: Some(gen11_dsi_host_attach), detach: Some(gen11_dsi_host_detach), transfer: Some(gen11_dsi_host_transfer) });

fn icl_dphy_param_init(intel_dsi: *mut intel_dsi) {
let connector: *mut intel_connector = unsafe { (*intel_dsi).attached_connector };
let mipi_config: *mut mipi_config = unsafe { (*connector).panel.vbt.dsi.config };
let tlpx_ns: u32 = (unsafe { intel_dsi_tlpx_ns(intel_dsi as *const intel_dsi) }) as u32;
let mut tclk_prepare_esc_clk: u32 = 0;
let mut tclk_zero_esc_clk: u32 = 0;
let mut tclk_pre_esc_clk: u32 = 0;
let mut ths_prepare_esc_clk: u32 = 0;
let mut ths_zero_esc_clk: u32 = 0;
let mut ths_exit_esc_clk: u32 = 0;
// The clock and data lane prepare timing parameters are in expressed in
// units of 1/4 escape clocks, and all the other timings parameters in
// escape clocks.
tclk_prepare_esc_clk = (((unsafe { (*mipi_config).tclk_prepare }) as i32).wrapping_mul(4) as u32).wrapping_add(tlpx_ns).wrapping_sub(1) / tlpx_ns;
let mut __v0: u32;
let __UNIQUE_ID_x__718: u32 = tclk_prepare_esc_clk;
let __UNIQUE_ID_y__719: i32 = 7;
loop {
if __types_ok__4!(__UNIQUE_ID_x__718, __UNIQUE_ID_y__719) == 0 {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__718 < (__UNIQUE_ID_y__719 as u32) { __UNIQUE_ID_x__718 } else { __UNIQUE_ID_y__719 as u32 };
tclk_prepare_esc_clk = __v0;
tclk_zero_esc_clk = (((unsafe { (*mipi_config).tclk_prepare_clkzero }) as i32).wrapping_sub((unsafe { (*mipi_config).tclk_prepare }) as i32) as u32).wrapping_add(tlpx_ns).wrapping_sub(1) / tlpx_ns;
let mut __v2: u32;
let __UNIQUE_ID_x__721: u32 = tclk_zero_esc_clk;
let __UNIQUE_ID_y__722: i32 = 15;
loop {
if __types_ok__4!(__UNIQUE_ID_x__721, __UNIQUE_ID_y__722) == 0 {
unsafe { __compiletime_assert_723() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__721 < (__UNIQUE_ID_y__722 as u32) { __UNIQUE_ID_x__721 } else { __UNIQUE_ID_y__722 as u32 };
tclk_zero_esc_clk = __v2;
tclk_pre_esc_clk = (((unsafe { (*mipi_config).tclk_pre }) as i32) as u32).wrapping_add(tlpx_ns).wrapping_sub(1) / tlpx_ns;
let mut __v4: u32;
let __UNIQUE_ID_x__724: u32 = tclk_pre_esc_clk;
let __UNIQUE_ID_y__725: i32 = 3;
loop {
if __types_ok__4!(__UNIQUE_ID_x__724, __UNIQUE_ID_y__725) == 0 {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__724 < (__UNIQUE_ID_y__725 as u32) { __UNIQUE_ID_x__724 } else { __UNIQUE_ID_y__725 as u32 };
tclk_pre_esc_clk = __v4;
ths_prepare_esc_clk = (((unsafe { (*mipi_config).ths_prepare }) as i32).wrapping_mul(4) as u32).wrapping_add(tlpx_ns).wrapping_sub(1) / tlpx_ns;
let mut __v6: u32;
let __UNIQUE_ID_x__727: u32 = ths_prepare_esc_clk;
let __UNIQUE_ID_y__728: i32 = 7;
loop {
if __types_ok__4!(__UNIQUE_ID_x__727, __UNIQUE_ID_y__728) == 0 {
unsafe { __compiletime_assert_729() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_x__727 < (__UNIQUE_ID_y__728 as u32) { __UNIQUE_ID_x__727 } else { __UNIQUE_ID_y__728 as u32 };
ths_prepare_esc_clk = __v6;
ths_zero_esc_clk = (((unsafe { (*mipi_config).ths_prepare_hszero }) as i32).wrapping_sub((unsafe { (*mipi_config).ths_prepare }) as i32) as u32).wrapping_add(tlpx_ns).wrapping_sub(1) / tlpx_ns;
let mut __v8: u32;
let __UNIQUE_ID_x__730: u32 = ths_zero_esc_clk;
let __UNIQUE_ID_y__731: i32 = 15;
loop {
if __types_ok__4!(__UNIQUE_ID_x__730, __UNIQUE_ID_y__731) == 0 {
unsafe { __compiletime_assert_732() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = if __UNIQUE_ID_x__730 < (__UNIQUE_ID_y__731 as u32) { __UNIQUE_ID_x__730 } else { __UNIQUE_ID_y__731 as u32 };
ths_zero_esc_clk = __v8;
ths_exit_esc_clk = (((unsafe { (*mipi_config).ths_exit }) as i32) as u32).wrapping_add(tlpx_ns).wrapping_sub(1) / tlpx_ns;
let mut __v10: u32;
let __UNIQUE_ID_x__733: u32 = ths_exit_esc_clk;
let __UNIQUE_ID_y__734: i32 = 7;
loop {
if __types_ok__4!(__UNIQUE_ID_x__733, __UNIQUE_ID_y__734) == 0 {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = if __UNIQUE_ID_x__733 < (__UNIQUE_ID_y__734 as u32) { __UNIQUE_ID_x__733 } else { __UNIQUE_ID_y__734 as u32 };
ths_exit_esc_clk = __v10;
// clock lane dphy timings
unsafe {
(*intel_dsi).dphy_reg = 2147483648u32 | CLK_PREPARE!(tclk_prepare_esc_clk) | 134217728u32 | CLK_ZERO!(tclk_zero_esc_clk) | 524288u32 | CLK_PRE!(tclk_pre_esc_clk);
}
// data lanes dphy timings
unsafe {
(*intel_dsi).dphy_data_lane_reg = 2147483648u32 | HS_PREPARE!(ths_prepare_esc_clk) | 8388608u32 | HS_ZERO!(ths_zero_esc_clk) | 128u32 | HS_EXIT!(ths_exit_esc_clk);
}
unsafe { intel_dsi_log_params(intel_dsi) };
}

fn icl_dsi_add_properties(connector: *mut intel_connector) {
let fixed_mode: *const drm_display_mode = unsafe { intel_panel_preferred_fixed_mode(connector) };
unsafe { intel_attach_scaling_mode_property(core::ptr::addr_of_mut!((*connector).base)) };
unsafe { drm_connector_set_panel_orientation_with_quirk(core::ptr::addr_of_mut!((*connector).base), intel_dsi_get_panel_orientation(connector), (*fixed_mode).hdisplay as i32, (*fixed_mode).vdisplay as i32) };
}

#[unsafe(no_mangle)]
extern "C" fn icl_dsi_init(display: *mut intel_display, devdata: *const intel_bios_encoder_data) {
let mut __slate_compound_36: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_38: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut host: core::mem::MaybeUninit<*mut intel_dsi_host> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut intel_dsi_host = core::ptr::addr_of_mut!(host) as *mut *mut intel_dsi_host;
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v8: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v8) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v17: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v17) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut port: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut i32 = core::ptr::addr_of_mut!(port) as *mut i32;
let mut connector: core::mem::MaybeUninit<*mut drm_connector> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut drm_connector = core::ptr::addr_of_mut!(connector) as *mut *mut drm_connector;
let mut intel_connector: core::mem::MaybeUninit<*mut intel_connector> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut intel_connector = core::ptr::addr_of_mut!(intel_connector) as *mut *mut intel_connector;
let mut encoder: core::mem::MaybeUninit<*mut intel_encoder> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut intel_encoder = core::ptr::addr_of_mut!(encoder) as *mut *mut intel_encoder;
let mut intel_dsi: core::mem::MaybeUninit<*mut intel_dsi> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut intel_dsi = core::ptr::addr_of_mut!(intel_dsi) as *mut *mut intel_dsi;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
*__slate_slot_27 = intel_bios_encoder_port(devdata);
if (*__slate_slot_27 as i32) == -1 {
return;
} else {
__obj_size = size_mul(core::mem::size_of::<intel_dsi>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_21 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_22 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_36 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_36)
});
alloc_tag_restore!();
} else {
*__slate_slot_22 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_38 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_38)
});
}
*__slate_slot_31 = *__slate_slot_22 as *mut intel_dsi;
if (*__slate_slot_31).is_null() {
return;
} else {
*__slate_slot_29 = intel_connector_alloc();
if (*__slate_slot_29).is_null() {
kfree(*__slate_slot_31 as *const ());
return;
} else {
'__join_0: {
*__slate_slot_30 = core::ptr::addr_of_mut!((*(*__slate_slot_31)).base);
(*(*__slate_slot_31)).attached_connector = *__slate_slot_29;
*__slate_slot_28 = core::ptr::addr_of_mut!((*(*__slate_slot_29)).base);
(*(*__slate_slot_30)).devdata = devdata;
// register DSI encoder with DRM subsystem
drm_encoder_init((*display).drm, core::ptr::addr_of_mut!((*(*__slate_slot_30)).base), core::ptr::addr_of!(gen11_dsi_encoder_funcs.0), 6i32, c"DSI %c".as_ptr() as *const u8, port_name!(*__slate_slot_27));
(*(*__slate_slot_30)).pre_pll_enable = Some(gen11_dsi_pre_pll_enable);
(*(*__slate_slot_30)).pre_enable = Some(gen11_dsi_pre_enable);
(*(*__slate_slot_30)).enable = Some(gen11_dsi_enable);
(*(*__slate_slot_30)).disable = Some(gen11_dsi_disable);
(*(*__slate_slot_30)).post_disable = Some(gen11_dsi_post_disable);
(*(*__slate_slot_30)).port = *__slate_slot_27;
(*(*__slate_slot_30)).get_config = Some(gen11_dsi_get_config);
(*(*__slate_slot_30)).sync_state = Some(gen11_dsi_sync_state);
(*(*__slate_slot_30)).update_pipe = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>>(intel_backlight_update as *const ()) };
(*(*__slate_slot_30)).compute_config = Some(gen11_dsi_compute_config);
(*(*__slate_slot_30)).get_hw_state = Some(gen11_dsi_get_hw_state);
(*(*__slate_slot_30)).initial_fastset_check = Some(gen11_dsi_initial_fastset_check);
(*(*__slate_slot_30)).r#type = 9;
(*(*__slate_slot_30)).cloneable = 0;
(*(*__slate_slot_30)).pipe_mask = 255u8;
(*(*__slate_slot_30)).power_domain = 41;
(*(*__slate_slot_30)).get_power_domains = Some(gen11_dsi_get_power_domains);
(*(*__slate_slot_30)).disable_clock = Some(gen11_dsi_gate_clocks);
(*(*__slate_slot_30)).is_clock_enabled = Some(gen11_dsi_is_clock_enabled);
(*(*__slate_slot_30)).shutdown = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut intel_encoder)>>(intel_dsi_shutdown as *const ()) };
// register DSI connector with DRM subsystem
drm_connector_init((*display).drm, *__slate_slot_28, core::ptr::addr_of!(gen11_dsi_connector_funcs.0), 16i32);
drm_connector_helper_add(*__slate_slot_28, core::ptr::addr_of!(gen11_dsi_connector_helper_funcs.0));
(*(*__slate_slot_28)).display_info.subpixel_order = 1;
(*(*__slate_slot_29)).get_hw_state = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut intel_connector) -> bool>>(intel_connector_get_hw_state as *const ()) };
// attach connector to encoder
intel_connector_attach_encoder(*__slate_slot_29, *__slate_slot_30);
(*(*__slate_slot_31)).panel_power_off_time = ktime_get_boottime();
intel_bios_init_panel_late(display, core::ptr::addr_of_mut!((*(*__slate_slot_29)).panel), (*(*__slate_slot_30)).devdata, core::ptr::null::<drm_edid>());
mutex_lock(core::ptr::addr_of_mut!((*(*display).drm).mode_config.mutex));
intel_panel_add_vbt_lfp_fixed_mode(*__slate_slot_29);
mutex_unlock(core::ptr::addr_of_mut!((*(*display).drm).mode_config.mutex));
if intel_panel_preferred_fixed_mode(*__slate_slot_29).is_null() {
while false {
}
_dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* DSI fixed mode info missing\n".as_ptr() as *const u8);
} else {
intel_panel_init(*__slate_slot_29, core::ptr::null::<drm_edid>());
intel_backlight_setup(*__slate_slot_29, -1i32);
if (core::ptr::read_unaligned(core::ptr::addr_of!((*(*(*__slate_slot_29)).panel.vbt.dsi.config).__slate_anon_2.__slate_bits_0)).__get_dual_link() as i32) != 0 {
(*(*__slate_slot_31)).__slate_anon_6.ports = (BIT!(0i32) | BIT!(1i32)) as u16;
} else {
(*(*__slate_slot_31)).__slate_anon_6.ports = BIT__2!(*__slate_slot_27) as u16;
}
__ret_warn_on_1 = (((*(*__slate_slot_29)).panel.vbt.dsi.bl_ports as i32) & !((*(*__slate_slot_31)).__slate_anon_6.ports as i32) != 0) as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_connector->panel.vbt.dsi.bl_ports & ~intel_dsi->ports)".as_ptr() as *mut u8);
*__slate_slot_17 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t1: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t1, sym __slate_asm_string_icl_dsi_init_34, sym __slate_asm_string_icl_dsi_init_35, const 2013 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_15 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t1 as u64) };
(*__slate_slot_17).unwrap()(*__slate_slot_15, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_connector->panel.vbt.dsi.bl_ports & ~intel_dsi->ports)".as_ptr() as *mut u8);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
if __ret_warn_on_1 != 0 {
let __v14: *mut intel_connector = *__slate_slot_29;
(*__v14).panel.vbt.dsi.bl_ports = (((*__v14).panel.vbt.dsi.bl_ports as i32) & ((*(*__slate_slot_31)).__slate_anon_6.ports as i32)) as u16;
}
__ret_warn_on = (((*(*__slate_slot_29)).panel.vbt.dsi.cabc_ports as i32) & !((*(*__slate_slot_31)).__slate_anon_6.ports as i32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_connector->panel.vbt.dsi.cabc_ports & ~intel_dsi->ports)".as_ptr() as *mut u8);
*__slate_slot_8 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_icl_dsi_init_32, sym __slate_asm_string_icl_dsi_init_33, const 2016 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_6 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_8).unwrap()(*__slate_slot_6, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(intel_connector->panel.vbt.dsi.cabc_ports & ~intel_dsi->ports)".as_ptr() as *mut u8);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
if __ret_warn_on != 0 {
let __v5: *mut intel_connector = *__slate_slot_29;
(*__v5).panel.vbt.dsi.cabc_ports = (((*__v5).panel.vbt.dsi.cabc_ports as i32) & ((*(*__slate_slot_31)).__slate_anon_6.ports as i32)) as u16;
}
*__slate_slot_27 = 0;
while (*__slate_slot_27 as i32) < 9 {
if (((*(*__slate_slot_31)).__slate_anon_6.ports as i32) as u64) & BIT__2!(*__slate_slot_27) != 0 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = intel_dsi_host_init(*__slate_slot_31, core::ptr::addr_of!(gen11_dsi_host_ops.0), *__slate_slot_27);
if (*__slate_slot_2).is_null() {
break '__join_0;
} else {
*((*(*__slate_slot_31)).dsi_hosts.as_mut_ptr() as *mut *mut intel_dsi_host).wrapping_offset((*__slate_slot_27 as i32) as isize) = *__slate_slot_2;
}
}
*__slate_slot_27 = (*__slate_slot_27 as i32).wrapping_add(1) as i32;
}
if !intel_dsi_vbt_init(*__slate_slot_31, 1u16) {
__drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"no device found\n".as_ptr() as *const u8);
} else {
icl_dphy_param_init(*__slate_slot_31);
icl_dsi_add_properties(*__slate_slot_29);
return;
}
}
}
drm_connector_cleanup(*__slate_slot_28);
drm_encoder_cleanup(core::ptr::addr_of_mut!((*(*__slate_slot_30)).base));
kfree(*__slate_slot_31 as *const ());
kfree(*__slate_slot_29 as *const ());
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_736: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_737: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_gen11_dsi_configure_transcoder_39: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_gen11_dsi_configure_transcoder_40: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_configure_transcoder_61: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_gen11_dsi_configure_transcoder_62: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_configure_transcoder_93: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_gen11_dsi_configure_transcoder_94: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_10: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_11: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_16: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_17: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_22: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_23: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_28: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_29: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_gen11_dsi_dsc_compute_config_5: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_get_dsi_io_power_domains_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_get_dsi_io_power_domains_5: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_icl_dsi_init_32: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_icl_dsi_init_33: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";

static __slate_asm_string_icl_dsi_init_34: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_icl_dsi_init_35: [u8; 68] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/icl_dsi.c\0";
