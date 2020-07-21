#ifndef __LIBUBLASAUX_MATRIXNICEOUTPUTER_H__
#define __LIBUBLASAUX_MATRIXNICEOUTPUTER_H__

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
#include <vector>
#include <boost/format.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace boost { namespace numeric { namespace ublas {


/**
 * Functor realizing nice & suitable for user stream output matrices.
 * @author Anton Liaukevich
 */
class MatrixNiceOutputer: public BaseNiceOutputer {
public:
    /* Types */

    enum ElementPlacing { SIMPLE, BY_COLUMNS, BY_EQUALWIDTH_COLUMNS };

    /* Construct/copy/destruct */

    /**
     * Creates a new functor for matrix outputing.
     * @param placing Strategy of table outputing @see ElementPlacing
     * @param minSpaces Minimal number of spaces adjacent columns (elements) separated by
     * @param isLineFeedAfterAll If true puts line feed after all outputed numbers
     */
    inline explicit MatrixNiceOutputer(ElementPlacing placing, StreamSize minSpaces = 1,
                                       bool isLineFeedAfterAll = true):
        BaseNiceOutputer(minSpaces, isLineFeedAfterAll), placing_(placing) {}

    //TODO: Is it must be virtual or non-virtual?
    inline ~MatrixNiceOutputer() {}

public:
    /* Field (read-only) access */

    inline ElementPlacing getPlacing() const
    {
        return placing_;
    }

    /* Real actions */

    /**
     * Outputs a matrix to the stream in a nice look. This is a dispatcher method which executes
     * private methods that implements output for concrete strategy specified by user in
     * constructor.
     * @param output Output stream
     * @param matrix A matrix object to be outputed. Can be object of any matrix type from the
     * boost::numeric::ublas library
     */
    template<class Char, class CharTraits, class Matrix>
    void operator()(std::basic_ostream<Char,CharTraits>& output, const Matrix& matrix) const
    {
        /* Output matrix sizes, new line after them */
        typedef boost::basic_format<Char,CharTraits> Format;
        output << Format("[%1%, %2%]\n") % matrix.size1() % matrix.size2();

        if (matrix.size1() == 0)
            output << "()";
        else if (getPlacing() == SIMPLE)
            doSimply(output, matrix);
        else if (getPlacing() == BY_COLUMNS)
            doJustifiedColumns(output, matrix);
        else if (getPlacing() == BY_EQUALWIDTH_COLUMNS)
            doEqualWidthColumns(output, matrix);

        if (isLineFeedAfterAll())
            output << "\n";
    }

private:
    /* Auxiliary methods */

    template<class Char, class CharTraits, class Matrix>
    void doSimply(std::basic_ostream<Char,CharTraits>& output, const Matrix& matrix) const
    {
        for (typename Matrix::size_type i = 0; i < matrix.size1(); ++i)
        {
            if (i == 0)
                output << "(";
            else
                output << " ";

            outputRowSimply(output, row(matrix, i));

            if (i + 1 == matrix.size1()) // cannot use "i == matrix.size1()-1" because Size may be unsigned
                output << ")";
            else
                output << ",\n";
        }
    }

    template<class Char, class CharTraits, class Matrix>
    void doJustifiedColumns(std::basic_ostream<Char,CharTraits>& output, const Matrix& matrix) const
    {
        typedef typename Matrix::size_type Size;
        Size m = matrix.size1(),
             n = matrix.size2();

        boost::numeric::ublas::matrix<StreamSize> elementOutputSizes(m, n);
        std::vector<StreamSize> columnWidths(n);
        for (Size i = 0; i < m; ++i)
            for (Size j = 0; j < n; ++j)
            {
                StreamSize current = countValueOutputSize(output, matrix(i, j));
                elementOutputSizes(i, j) = current;
                if (current > columnWidths[j])
                    columnWidths[j] = current;
            }

        for (Size i = 0; i < m; ++i)
        {
            if (i == 0)
                output << "(";
            else
                output << " ";

            output << "(";
            for (Size j = 0; j + 1 < n; ++j) // cannot use "j < n-1" because Size may be unsigned
                output << matrix(i, j) << ","
                       << spacesNeeded<Char,CharTraits>(elementOutputSizes(i, j), columnWidths[j]);
            if (n >= 1)
                output << matrix(i, n-1)
                       << std::basic_string<Char,CharTraits>
                           (columnWidths[n-1] - elementOutputSizes(i, n-1), ' ');
            output << ")";

            if (i + 1 == m) // cannot use "i == m-1" because Size may be unsigned
                output << ")";
            else
                output << ",\n";
        }
    }

    template<class Char, class CharTraits, class Matrix>
    void doEqualWidthColumns(std::basic_ostream<Char,CharTraits>& output, const Matrix& matrix) const
    {
        typedef typename Matrix::size_type Size;
        Size m = matrix.size1(),
             n = matrix.size2();

        boost::numeric::ublas::matrix<StreamSize> elementOutputSizes(m, n);
        StreamSize width     = 0,
                   lastWidth = 0;
        for (Size i = 0; i < m; ++i)
            for (Size j = 0; j < n; ++j)
            {
                StreamSize current = countValueOutputSize(output, matrix(i, j));
                elementOutputSizes(i, j) = current;
                if (current > width)
                    width = current;
            }
        if (n > 0)
            for (Size i = 0; i < m; ++i)
                if (elementOutputSizes(i, n-1) > lastWidth)
                    lastWidth = elementOutputSizes(i, n-1);

        for (Size i = 0; i < m; ++i)
        {
            if (i == 0)
                output << "(";
            else
                output << " ";

            output << "(";
            for (Size j = 0; j + 1 < n; ++j) // cannot use "j < n-1" because Size may be unsigned
                output << matrix(i, j) << ","
                       << spacesNeeded<Char,CharTraits>(elementOutputSizes(i, j), width);
            if (n >= 1)
                output << matrix(i, n-1)
                       << std::basic_string<Char,CharTraits>(width - elementOutputSizes(i, n-1), ' ');
            output << ")";

            if (i + 1 == m) // cannot use "i == m-1" because Size may be unsigned
                output << ")";
            else
                output << ",\n";
        }
    }

    /**
     * Auxiliary method.
     * @return String consisted of spaces necessary to justify column.
     */
    template<class Char, class CharTraits>
    inline std::basic_string<Char,CharTraits> spacesNeeded(StreamSize itemOutputSize,
                                                           StreamSize justifySize) const
    {
        StreamSize count = justifySize - itemOutputSize + minSpaces_;
        return std::basic_string<Char,CharTraits>(count, ' ');
    }

    /* Fields */

    ElementPlacing placing_;

}; //class MatrixNiceOutputer


}}} //namespace boost::numeric::ublas

#endif //__LIBUBLASAUX_MATRIXNICEOUTPUTER_H__
