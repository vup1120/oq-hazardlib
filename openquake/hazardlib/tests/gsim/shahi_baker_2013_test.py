# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2013-2016 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

from openquake.hazardlib.gsim.shahi_baker_2013 import ShahiBaker2013

from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase

# Test data have been generated frmo the implemented GMPE itself
# since the reference code is not available from the author


class ShahiBaker2013TestCase(BaseGSIMTestCase):
    GSIM_CLASS = ShahiBaker2013

    def test_mean_strikeslip_faulting(self):
        self.check('CBR13/CBR13_SS_MEAN.csv', max_discrep_percentage=0.1)

    def test_mean_reverse_faulting(self):
        self.check('CBR13/CBR13_RV_MEAN.csv', max_discrep_percentage=0.1)

    def test_std_total_reverse(self):
        self.check('CBR13/CBR13_RV_STD_TOTAL.csv', max_discrep_percentage=0.1)

    def test_std_total_strikeslip(self):
        self.check('CBR13/CBR13_SS_STD_TOTAL.csv', max_discrep_percentage=0.1)
