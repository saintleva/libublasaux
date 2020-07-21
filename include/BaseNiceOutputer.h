#ifndef __LIBUBLASAUX_BASENICEOUTPUTER_H__
#define __LIBUBLASAUX_BASENICEOUTPUTER_H__

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

#include <ios>
#include <string>
#include <sstream>

namespace boost { namespace numeric { namespace ublas {


/**
 * Base class for "NiceOutputer"s - functors realizing nice & suitable stream output of vectors and
 * matrices. This base class contains common (for "NiceOutputer"s) properties and auxiliary methods.
 * User must use its descendants: VectorNiceOutputer and MatrixNiceOutputer.
 * @author Anton Liaukevich
 * @brief Base class for VectorNiceOutputer and MatrixNiceOutputer.
 */
class BaseNiceOutputer {
public:
    /* Types */

    /**
     * Integer type used to count chars in i/o streams of STL library.
     */
    typedef std::streamsize StreamSize;

protected:
    /* Construct/copy/destruct */

    inline BaseNiceOutputer(StreamSize minSpaces, bool isLineFeedAfterAll):
        minSpaces_(minSpaces), isLineFeedAfterAll_(isLineFeedAfterAll) {}

    //TODO: Is it must be public or protected, virtual or non-virtual?
    inline ~BaseNiceOutputer() {}

public:
    /* Field (read-only) access */

    inline StreamSize getMinSpaces() const
    {
        return minSpaces_;
    }

    inline bool isLineFeedAfterAll() const
    {
        return isLineFeedAfterAll_;
    }

protected:

    /**
     * Auxiliary method. Outputs a vector in a simple way (non-justified) to stream.
     * @param output Output stream
     * @param vector Vector be outputed
     */
    template<class Char, class CharTraits, class Vector>
    void outputRowSimply(std::basic_ostream<Char,CharTraits>& output, const Vector& vector) const
    {
        typedef typename Vector::size_type Size;

        output << "(";

        Size size = vector.size();
        for (Size i = 0; i + 1 < size; ++i) // cannot use "i < size-1" because Size may be unsigned
            output << vector(i) << "," << std::basic_string<Char,CharTraits>(getMinSpaces(), ' ');
        if (size >= 1)
            output << vector(size - 1);

        output << ")";
    }

    /**
     * Auxiliary function. Calculate length of text that will be stream output of a given value.
     * Temporary string stream is used to do this.
     * @param output Output stream
     * @param value Value to be outputed
     */
    template<class Char, class CharTraits, class Value>
    static StreamSize countValueOutputSize(const std::basic_ostream<Char,CharTraits>& output,
                                           const Value& value)
    {
        std::basic_ostringstream<Char, CharTraits, std::allocator<Char> > temp;
        temp.flags(output.flags());
        temp.imbue(output.getloc());
        temp.precision(output.precision());

        temp << value;
        return temp.str().size();
    }

protected:
    /* Fields */

    StreamSize minSpaces_;
    bool isLineFeedAfterAll_;

}; //class BaseNiceOutputer


}}} //namespace boost::numeric::ublas

#endif //__LIBUBLASAUX_BASENICEOUTPUTER_H__
