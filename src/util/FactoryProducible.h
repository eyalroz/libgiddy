#ifndef UTIL_FACTORY_PRODUCIBLE_H_
#define UTIL_FACTORY_PRODUCIBLE_H_

#include <typeinfo>
 
#include "util/Factory.h"
#include "util/type_name.hpp"
#include <string>
#include <exception>
#include <memory>

namespace util {

namespace mixins {

#ifndef UTIL_EXCEPTION_H_
using std::logic_error;
#endif

/**
 * Note: This is a mix-in class, and you can't actually instantiate it!
 * I mean, maybe you sort of can, but it won't work unless you
 * implement the main "pseudo-virtual" method
 */
template <typename Key, typename Base, typename... ConstructionArgs>
class FactoryProducible {
public:

	template <typename U>
	static void registerInSubclassFactory(const Key& key, bool ignore_repeat_registration = true) {
		getSubclassFactory_().template registerClass<U>(key, ignore_repeat_registration);
	}

protected:
	using SubclassFactory = util::ExposedFactory<Key, Base, ConstructionArgs...>;

	/**
	 * This method is sort of an attempt to avoid the static initialization
	 * fiasco; if you call it, you're guaranteed that the initialization
	 * of subclass_factory happens before you get it.
	 *
	 * @return the class' static factory for producing subclasses - initialized
	 */

	static SubclassFactory& getSubclassFactory_() {
		static SubclassFactory subclass_factory;
		return subclass_factory;
	}


public:
	static const SubclassFactory& getSubclassFactory() {
		return getSubclassFactory_();
	}

	// This is not implemented generically for the mixin class, which
	// makes it a sort of a virtual static method - but virtual only in
	// the sense of the template arguments.
	static Key resolveSubclassKey(ConstructionArgs... args);


	// Can't I return an rvalue reference and avoid pointers altogether?
	static std::unique_ptr<Base> produceSubclass(const Key& subclass_key, ConstructionArgs... args) {
		if (not getSubclassFactory().canProduce(subclass_key)) {
			throw std::invalid_argument(std::string("No subclass of the base type ")
				+ util::type_name<Base>() + " is registered with key \""
				+ std::string(subclass_key) + "\"");
		}
		return getSubclassFactory().produce(subclass_key, args...);
	}

	static std::unique_ptr<Base> produceSubclass(ConstructionArgs... args) {
		Key subclass_key = resolveSubclassKey(args...);
		return produceSubclass(subclass_key, args...);
	}

	static bool canProduce(ConstructionArgs... args) {
		Key subclass_key = resolveSubclassKey(args...);
		return getSubclassFactory().canProduce(subclass_key);
	}

	FactoryProducible(const FactoryProducible& other) = delete;
	FactoryProducible() = default;
	~FactoryProducible() = default;
};

} // namespace mixins
} // namespace util

#endif /* UTIL_FACTORY_PRODUCIBLE_H_ */
