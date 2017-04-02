#pragma once
#ifndef SRC_KERNELS_REDUCTION_COMMON_CUH_
#define SRC_KERNELS_REDUCTION_COMMON_CUH_

#ifdef __CUDACC__
#define __hfd__ __host__ __device__ __forceinline__
#else
#define __hfd__
#endif


namespace cuda {
namespace kernels {
namespace reduction {

/*
 * This is a slightly hairy SFINAE gadget for using the same code
 * with different kinds of "unary" operatiors:
 *
 * We want the same kernel code to be able to apply both "enumerated"
 * unary operations (which take a data element and its index) and
 * "unenumerated" ones (which only take the element) with the same
 * code. The selection is by template parameter, but we can't do
 *
 *    if (use_enumerated) { y = f(index, x); }
 *    else                { y = f(x);        }
 *
 * since both branches of the if statement must be able to
 * compile. Only in C++17 will a feature be introduced which reduces
 * the requirement for compile-time-resolvable if branches from
 * compilability to syntactic correctness. The most recent paper on
 * this seems to be:
 *
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0292r0.html
 *
 * For now, we use the gadget below to wrap invocations of both
 * enumerated and unenumerated functions, so the template and/or
 * overload resolution, instead of if-branch resolution, are what leads
 * us to the two different kinds of calls above.
 *
 * The only thing you need to remember is to call the "outer" function,
 * which is apply_unary_op without a tag argument, whenever you need
 * the application of your op (enumerated or not) on an argument, and
 * to always pass that argument's element index, even if it's not to be
 * used.
 *
 */
namespace detail {

template <typename, typename = void>
struct is_enumerated { enum { value = false }; };

template <typename T>
struct is_enumerated<
	T,
	typename std::enable_if<sizeof(typename T::enumerator_type)||true>::type>
{ enum { value = true }; };

struct unenumerated_tag {};
struct enumerated_tag {};

// TODO: Consider making this a specialty enum type - maybe the one
// aready defined in cuda/functors.hpp
template <bool IsEnumerated> struct traits;
template <> struct traits<false> {  using tag = unenumerated_tag; };
template <> struct traits<true>  {  using tag = enumerated_tag;   };

template <typename E, typename UnaryOp>
static __hfd__ typename UnaryOp::result_type apply_unary_op(
	const E& e, const typename UnaryOp::argument_type& x, const unenumerated_tag)
{
	return UnaryOp()(x);
}
template <typename E, typename EnumeratedUnaryOp>
static __hfd__ typename EnumeratedUnaryOp::result_type apply_unary_op(
	const E& e, const typename EnumeratedUnaryOp::argument_type& x, const enumerated_tag)
{
	return EnumeratedUnaryOp()(e, x);
}

template <typename E, typename PossiblyEnumeratedUnaryOp>
static __hfd__ typename PossiblyEnumeratedUnaryOp::result_type apply_unary_op(
	const E& e, const typename PossiblyEnumeratedUnaryOp::argument_type& x)
{
	return  apply_unary_op<E, PossiblyEnumeratedUnaryOp>(
		e, x, typename traits<is_enumerated<PossiblyEnumeratedUnaryOp>::value>::tag ());
}


} // namespace detail

} // namespace reduction
} // namespace kernels
} // namespace cuda


#endif /* SRC_KERNELS_REDUCTION_COMMON_CUH_ */
