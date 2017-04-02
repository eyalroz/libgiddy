/**
 * Implementation of the factory pattern, based on suggestions here:
 *
 * http://stackoverflow.com/q/5120768/1593077
 *
 * and on the suggestions for corrections here:
 *
 * http://stackoverflow.com/a/34948111/1593077
 *
 * which addresses the finer points of template + rvalue reference + move-vs-forward
 *
 */
#pragma once
#ifndef UTIL_FACTORY_H_
#define UTIL_FACTORY_H_

#include <unordered_map>
#include <memory>
#include <exception>


namespace util {

#ifndef UTIL_EXCEPTION_H_
using std::logic_error;
#endif

/*
 * TODO:
 * - Add a template parameter for choosing pointer return policy: raw vs unique_ptr
 * - Perhaps we don't really need createInstance, and should just use a labmda?
 * - This might be made even more generic by factoring out the is_base_of check, making
 *   the class a general-purpose factory, calling a templated compatibility check
 */

template<typename Key, typename T, typename... ConstructionArgs>
class Factory {
public:
	using Instantiator = T* (*)(ConstructionArgs...);
protected:
	template<typename U>
	static T* createInstance(ConstructionArgs&&... args)
	{
		return new U(std::forward<ConstructionArgs>(args)...);
	}
	using Instantiators = std::unordered_map<Key,Instantiator>;


	Instantiators subclassInstantiators;

public:
	template<typename U>
	void registerClass(const Key& key, bool ignore_repeat_registration = false)
	{
		// TODO: - Consider repeat-registration behavior.
		static_assert(std::is_base_of<T, U>::value,
			"This factory cannot register a class which is is not actually "
			"derived from the factory's associated class");
		auto it = subclassInstantiators.find(key);
		if (it != subclassInstantiators.end()) {
			if (ignore_repeat_registration) { return; }
			throw logic_error("Repeat registration of the same subclass in this factory.");
		}
		subclassInstantiators.emplace(key, &createInstance<U>);
	}

public:
	// TODO: Consider marking this pointer non-null somehow
	std::unique_ptr<T> produce(const Key& subclass_key, ConstructionArgs... args) const
	{
		auto it = subclassInstantiators.find(subclass_key);
		if (it == subclassInstantiators.end()) {
			throw std::invalid_argument(
				"Attempt to produce an instance of a class "
				"by a key not registered in the factory");
		}
		auto instantiator = it->second;
		return std::unique_ptr<T>(instantiator(std::forward<ConstructionArgs>(args)...));
	}

	bool canProduce(const Key& subclass_key) const {
		return subclassInstantiators.find(subclass_key) != subclassInstantiators.end();
	}
};

/**
 * This is for when you want to be able to list factory contents. It's a bit ugly
 * but, hey, nobody said you have to use it.
 */
template<typename Key, typename T, typename... ConstructionArgs>
class ExposedFactory : public Factory<Key, T, ConstructionArgs...> {

	using parent = Factory<Key, T, ConstructionArgs...>;
	using Instantiator = typename parent::Instantiator;
	using Instantiators = typename parent::Instantiators;

public:
	const Instantiators& getInsantiators() const {
		return parent::subclassInstantiators;
	}
};

} // namespace util

#endif /* UTIL_FACTORY_H_ */
