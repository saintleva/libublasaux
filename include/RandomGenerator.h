#ifndef __LIBUBLASAUX_RANDOMGENERATOR_H__
#define __LIBUBLASAUX_RANDOMGENERATOR_H__

/*
 * Copyright (C) Anton Liaukevich 2006-2008 <leva.dev@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "StdDispatchRandomizer.h"
#include <boost/numeric/ublas/expression_types.hpp>

namespace boost { namespace numeric { namespace ublas {


struct EmptyType {
    typedef void Distribution;
};

/**
 * Strategy-driven functor (class template). It uses powerful "random" library of Boost
 * to random-fill of vectors and matrices.
 * @author Anton Liaukevich
 * @brief A simple frontend for random-filling of vectors and matrices.
 * @tparam Engine_ Engine for generating preudo-random numbers. Distribution object also required
 * for "variate_generator" in order to actually generate random numbers.
 * @tparam ItemDistribution_ Probability distribution for elements of vectors or matrices.
 * @tparam IndexDistributionCreator_ Probability distribution for is also required for indices
 * (integer types) in order to choose positions to place random values into them. This parameter is
 * not a distribution themself. It is a creator (similar to factory) and used to create real
 * distribution objects as of only integer parameter (size). Default value "EmptyType" does not
 * have such semantics therefore you will not be able to randomize sparse matrices in such case.
 * @tparam DispatchRandomizer Strategy used to dispatch factical randomizing to partial specializations
 * in order to implement different behaviour for different container types (class templates).
 * Default strategy "StdDispatchRandomizer" supports all matrix and vector types (templates) from
 * Boost::numeric::uBLAS library and randomizes them without errors & waste of time.
 * @remark I think (as contrasted with Andrei Alexandrescu) protected inheritance from
 * strategy class to be enough here.
 */
template<
         class Engine_,
         class ItemDistribution_,
         class IndexDistributionCreator_ = EmptyType,
         template<class,class,class> class DispatchRandomizer = StdDispatchRandomizer
        >
class RandomGenerator:
        protected StdDispatchRandomizer<Engine_,ItemDistribution_,IndexDistributionCreator_> {
public:
    /* Types */

    typedef Engine_ Engine;
    typedef ItemDistribution_ ItemDistribution;
    typedef IndexDistributionCreator_ IndexDistributionCreator;

    /* Construct/copy/destruct */

    /**
     * Constructs an object of "RandomGenerator" by given random-generator engine (non-const reference)
     * and probability distribution for elements.
     * @brief Basic constructor.
     * @param engine Engine for generating random (or preudo-random) numbers.
     * @param itemDistribution Probability distribution of random numbers that is used to fill vectors &
     * matrices.
     */
    inline RandomGenerator(Engine& engine, const ItemDistribution& itemDistribution):
        engine_(engine), itemDistribution_(itemDistribution) {}

    /* Real actions */

    /**
     * Fills user's contatiner with random numbers. This is main function and user have to use it to
     * randomize vectors & matrices.
     * @tparam Container Vector or matrix type. It is deduced from function argument.
     * @param[out] container Non-const reference to vector or matrix.
     */
    template<class Container>
    inline void operator()(Container& container) const
    {
        randomize(container, engine_, itemDistribution_);
    }

    /* Field (random-backend) (read-only) access */

    /**
     * @return Pointer to random-generating engine used here.
     */
    inline Engine* getEngine() const
    {
        return &engine_;
    }

    /**
     * @return Probability distribution for container elements.
     */
    inline ItemDistribution getItemDistribution() const
    {
        return itemDistribution_;
    }

private:
    /* Fields */

    Engine& engine_;
    ItemDistribution itemDistribution_;

}; //class RandomGenerator

/**
 * Function for convenient creation of RandomGenerator functor instance. You don't need to
 * explicitly indicate template parameters.
 * @see RandomGenerator
 * @author Anton Liaukevich
 * @return Object of "RandomGenerator<BaseGenerator,Distribution>" type with given
 * generator and distribution
 */
template<class Engine, class ItemDistribution>
RandomGenerator<Engine,ItemDistribution>
makeSimpleRandomGenerator(Engine& engine, const ItemDistribution& itemDistribution)
{
    return RandomGenerator<Engine,ItemDistribution>(engine, itemDistribution);
}


}}} //namespace boost::numeric::ublas

#endif // __LIBUBLASAUX_RANDOMGENERATOR_H__
