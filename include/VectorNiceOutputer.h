#ifndef __LIBUBLASAUX_VECTORNICEOUTPUTER_H__
#define __LIBUBLASAUX_VECTORNICEOUTPUTER_H__

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

#include "BaseNiceOutputer.h"
#include <ostream>
#include <boost/format.hpp>

namespace boost { namespace numeric { namespace ublas {


/**
 * Functor realizing nice & suitable for user stream output of vectors.
 * @author Anton Liaukevich
 */
class VectorNiceOutputer: public BaseNiceOutputer {
public:
    /* Construct/copy/destruct */

    /**
     * Creates a new functor for vector outputing.
     * @param isLineFeedAfterSize If true puts line feed after vector size
     * @param minSpaces Minimal number of spaces adjacent columns (elements) separated by
     * @param isLineFeedAfterAll If true puts line feed after all outputed numbers
     */
    inline explicit VectorNiceOutputer(bool isLineFeedAfterSize, StreamSize minSpaces = 1,
                                       bool isLineFeedAfterAll = true):
        BaseNiceOutputer(minSpaces, isLineFeedAfterAll), isLineFeedAfterSize_(isLineFeedAfterSize) {}

    //TODO: Is it must be virtual or non-virtual?
    inline ~VectorNiceOutputer() {}

public:
    /* Field (read-only) access */

    inline bool isLineFeedAfterSize() const
    {
        return isLineFeedAfterSize_;
    }

    /* Real actions */

    /**
     * Outputs a vectror to the stream in a nice look.
     * @param output Output stream
     * @param vector A vector object to be outputed. Can be object of any vector type from the
     * boost::numeric::ublas library
     */
    template<class Char, class CharTraits, class Vector>
    void operator()(std::basic_ostream<Char,CharTraits>& output, const Vector& vector) const
    {
        typedef boost::basic_format<Char,CharTraits> Format;
        output << Format("[%1%]") % vector.size();
        if (isLineFeedAfterSize())
            output << "\n";

        outputRowSimply(output, vector);

        if (isLineFeedAfterAll())
            output << "\n";
    }

private:
    /* Fields */

    bool isLineFeedAfterSize_;

}; //class VectorNiceOutputer


}}} //namespace boost::numeric::ublas

#endif //__LIBUBLASAUX_VECTORNICEOUTPUTER_H__
