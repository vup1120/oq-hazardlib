#!/bin/bash
if [ -n "$GEM_SET_DEBUG" -a "$GEM_SET_DEBUG" != "false" ]; then
    export PS4='+${BASH_SOURCE}:${LINENO}:${FUNCNAME[0]}: '
    set -x
fi
set -e
GEM_GIT_REPO="git://github.com/gem"
GEM_GIT_PACKAGE="oq-hazardlib"
GEM_DEB_PACKAGE="python-${GEM_GIT_PACKAGE}"
GEM_DEB_SERIE="master"
if [ -z "$GEM_DEB_REPO" ]; then
    GEM_DEB_REPO="$HOME/gem_ubuntu_repo"
fi
if [ -z "$GEM_DEB_MONOTONE" ]; then
    GEM_DEB_MONOTONE="$HOME/monotone"
fi

GEM_BUILD_ROOT="build-deb"
GEM_BUILD_SRC="${GEM_BUILD_ROOT}/${GEM_DEB_PACKAGE}"

GEM_ALWAYS_YES=false

if [ "$GEM_EPHEM_CMD" = "" ]; then
    GEM_EPHEM_CMD="lxc-start-ephemeral"
fi
if [ "$GEM_EPHEM_NAME" = "" ]; then
    GEM_EPHEM_NAME="ubuntu-lxc-eph"
fi

if command -v lxc-shutdown &> /dev/null; then
    # Older lxc (< 1.0.0) with lxc-shutdown
    LXC_TERM="lxc-shutdown -t 10 -w"
    LXC_KILL="lxc-stop"
else
    # Newer lxc (>= 1.0.0) with lxc-stop only
    LXC_TERM="lxc-stop -t 10"
    LXC_KILL="lxc-stop -k"
fi

NL="
"
TB="	"

#
#  functions
sig_hand () {
    trap ERR
    echo "signal trapped"
    if [ "$lxc_name" != "" ]; then
        set +e
        echo "Destroying [$lxc_name] lxc"
        upper="$(mount | grep "${lxc_name}.*upperdir" | sed 's@.*upperdir=@@g;s@,.*@@g')"
        if [ -f "${upper}.dsk" ]; then
            loop_dev="$(sudo losetup -a | grep "(${upper}.dsk)$" | cut -d ':' -f1)"
        fi
        sudo $LXC_KILL -n $lxc_name
        sudo umount /var/lib/lxc/$lxc_name/rootfs
        sudo umount /var/lib/lxc/$lxc_name/ephemeralbind
        echo "$upper" | grep -q '^/tmp/'
        if [ $? -eq 0 ]; then
            sudo umount "$upper"
            sudo rm -r "$upper"
            if [ "$loop_dev" != "" ]; then
                sudo losetup -d "$loop_dev"
                if [ -f "${upper}.dsk" ]; then
                    sudo rm -f "${upper}.dsk"
                fi
            fi
        fi
        sudo lxc-destroy -n $lxc_name
    fi
    if [ -f /tmp/packager.eph.$$.log ]; then
        rm /tmp/packager.eph.$$.log
    fi
    exit 1
}

#
#  repo_id_get - retry git repo from local git remote command
repo_id_get () {
    local repo_name repo_line

    if ! repo_name="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)"; then
        repo_line="$(git remote -vv | grep "^origin[ ${TB}]" | grep '(fetch)$')"
        if [ -z "$repo_line" ]; then
            echo "no remote repository associated with the current branch, exit 1"
            exit 1
        fi
    else
        repo_name="$(echo "$repo_name" | sed 's@/.*@@g')"

        repo_line="$(git remote -vv | grep "^${repo_name}[ ${TB}].*(fetch)\$")"
    fi

    if echo "$repo_line" | grep -q '[0-9a-z_-\.]\+@[a-z0-9_-\.]\+:'; then
        repo_id="$(echo "$repo_line" | sed "s/^[^ ${TB}]\+[ ${TB}]\+[^ ${TB}@]\+@//g;s/.git[ ${TB}]\+(fetch)$/.git/g;s@/${GEM_GIT_PACKAGE}.git@@g;s@:@/@g")"
    else
        repo_id="$(echo "$repo_line" | sed "s/^[^ ${TB}]\+[ ${TB}]\+git:\/\///g;s/.git[ ${TB}]\+(fetch)$/.git/g;s@/${GEM_GIT_PACKAGE}.git@@g")"
    fi

    echo "$repo_id"
}

#
#  mksafedir <dname> - try to create a directory and rise an alert if it already exists
#      <dname>    name of the directory to create
#
mksafedir () {
    local dname

    dname="$1"
    if [ "$GEM_ALWAYS_YES" != "true" -a -d "$dname" ]; then
        echo "$dname already exists"
        echo "press Enter to continue or CTRL+C to abort"
        read a
    fi
    rm -rf $dname
    mkdir -p $dname
}

usage () {
    local ret

    ret=$1

    echo
    echo "USAGE:"
    echo "    $0 [<-s|--serie> <precise|trusty>] [-D|--development] [-S--sources_copy] [-B|--binaries] [-U|--unsigned] [-R|--repository]    build debian source package."
    echo "       if -s is present try to produce sources for a specific ubuntu version (precise or trusty),"
    echo "           (default precise)"
    echo "       if -S is present try to copy sources to <GEM_DEB_MONOTONE>/<BUILD_UBUVER>/source directory"
    echo "       if -B is present binary package is build too."
    echo "       if -R is present update the local repository to the new current package"
    echo "       if -D is present a package with self-computed version is produced."
    echo "       if -U is present no sign are perfomed using gpg key related to the mantainer."
    echo "    $0 pkgtest <branch-name>                    run packaging tests into an ubuntu lxc environment"
    echo "    $0 devtest <branch-name>                    run development tests into an ubuntu lxc environment"
    echo
    exit $ret
}

_wait_ssh () {
    local lxc_ip="$1"

    for i in $(seq 1 20); do
        if ssh $lxc_ip "echo begin"; then
            break
        fi
        sleep 2
    done
    if [ $i -eq 20 ]; then
        return 1
    fi
}

_pkgbuild_innervm_run () {
    local lxc_ip="$1"
    local DPBP_FLAG="$2"

    trap 'local LASTERR="$?" ; trap ERR ; (exit $LASTERR) ; return' ERR

    ssh $lxc_ip mkdir build-deb
    scp -r * $lxc_ip:build-deb
    gpg -a --export | ssh $lxc_ip "sudo apt-key add -"
    ssh $lxc_ip sudo apt-get update
    ssh $lxc_ip sudo apt-get -y upgrade
    ssh $lxc_ip sudo apt-get -y install build-essential dpatch fakeroot devscripts equivs lintian quilt
    ssh $lxc_ip "sudo mk-build-deps --install --tool 'apt-get -y' build-deb/debian/control"

    ssh $lxc_ip "cd build-deb && dpkg-buildpackage $DPBP_FLAG"
    scp -r $lxc_ip:*.{tar.gz,deb,changes,dsc} ../

    return
}

_devtest_innervm_run () {
    local i lxc_ip="$1"

    trap 'local LASTERR="$?" ; trap ERR ; (exit $LASTERR) ; return' ERR

    ssh $lxc_ip "sudo apt-get update"
    ssh $lxc_ip "sudo apt-get upgrade -y"

    gpg -a --export | ssh $lxc_ip "sudo apt-key add -"
    # install package to manage repository properly
    ssh $lxc_ip "sudo apt-get install -y python-software-properties"

    # add custom packages
    ssh $lxc_ip mkdir -p "repo"
    scp -r ${GEM_DEB_REPO}/${BUILD_UBUVER}/custom_pkgs $lxc_ip:repo/custom_pkgs
    ssh $lxc_ip "sudo apt-add-repository \"deb file:/home/ubuntu/repo/custom_pkgs ./\""

    ssh $lxc_ip "sudo apt-get update"
    ssh $lxc_ip "sudo apt-get upgrade -y"

    pkgs_list="$(deps_list all debian)"
    ssh $lxc_ip "sudo apt-get install -y ${pkgs_list}"

    # TODO: version check
    git archive --prefix ${GEM_GIT_PACKAGE}/ HEAD | ssh $lxc_ip "tar xv"

    if [ -z "$GEM_DEVTEST_SKIP_TESTS" ]; then
        if [ -n "$GEM_DEVTEST_SKIP_SLOW_TESTS" ]; then
            # skip slow tests
            skip_tests="!slow,"
        fi
        ssh $lxc_ip "cd $GEM_GIT_PACKAGE ; nosetests -v -a '${skip_tests}' --with-doctest --with-coverage --cover-package=openquake.hazardlib --with-xunit"
        scp "$lxc_ip:$GEM_GIT_PACKAGE/nosetests.xml" "out_${BUILD_UBUVER}/"
    else
        if [ -d $HOME/fake-data/$GEM_GIT_PACKAGE ]; then
            cp $HOME/fake-data/$GEM_GIT_PACKAGE/* "out_${BUILD_UBUVER}/"
        fi
    fi

    trap ERR

    return
}

_pkgtest_innervm_run () {
    local lxc_ip="$1"

    trap 'local LASTERR="$?" ; trap ERR ; (exit $LASTERR) ; return' ERR

    ssh $lxc_ip "sudo apt-get update"
    ssh $lxc_ip "sudo apt-get -y upgrade"
    gpg -a --export | ssh $lxc_ip "sudo apt-key add -"
    # install package to manage repository properly
    ssh $lxc_ip "sudo apt-get install -y python-software-properties"

    # create a remote "local repo" where place $GEM_DEB_PACKAGE package
    ssh $lxc_ip mkdir -p repo/${GEM_DEB_PACKAGE}
    scp build-deb/${GEM_DEB_PACKAGE}_*.deb build-deb/${GEM_DEB_PACKAGE}_*.changes \
        build-deb/${GEM_DEB_PACKAGE}_*.dsc build-deb/${GEM_DEB_PACKAGE}_*.tar.gz \
        build-deb/Packages* build-deb/Sources*  build-deb/Release* $lxc_ip:repo/${GEM_DEB_PACKAGE}
    ssh $lxc_ip "sudo apt-add-repository \"deb file:/home/ubuntu/repo/${GEM_DEB_PACKAGE} ./\""

    # add custom packages
    scp -r ${GEM_DEB_REPO}/${BUILD_UBUVER}/custom_pkgs $lxc_ip:repo/custom_pkgs
    ssh $lxc_ip "sudo apt-add-repository \"deb file:/home/ubuntu/repo/custom_pkgs ./\""

    ssh $lxc_ip "sudo apt-get update"
    ssh $lxc_ip "sudo apt-get upgrade -y"

    # packaging related tests (install, remove, purge, install, reinstall)
    ssh $lxc_ip "sudo apt-get install -y ${GEM_DEB_PACKAGE}"
    ssh $lxc_ip "sudo apt-get remove -y ${GEM_DEB_PACKAGE}"
    ssh $lxc_ip "sudo apt-get install -y ${GEM_DEB_PACKAGE}"
    ssh $lxc_ip "sudo apt-get install --reinstall -y ${GEM_DEB_PACKAGE}"

    scp -r "$lxc_ip://usr/share/doc/${GEM_DEB_PACKAGE}/changelog*" .
    scp -r "$lxc_ip://usr/share/doc/${GEM_DEB_PACKAGE}/README*" .

    trap ERR

    return
}

_builddoc_innervm_run () {
    local i lxc_ip="$1"

    trap 'local LASTERR="$?" ; trap ERR ; (exit $LASTERR) ; return' ERR

    ssh $lxc_ip "sudo apt-get update"
    ssh $lxc_ip "sudo apt-get upgrade -y"

    gpg -a --export | ssh $lxc_ip "sudo apt-key add -"
    # install package to manage repository properly
    # ssh $lxc_ip "sudo apt-get install -y python-software-properties"

    pkgs_list="$(deps_list all debian)"
    ssh $lxc_ip "sudo apt-get install -y ${pkgs_list}"

    # TODO: version check
    git archive --prefix ${GEM_GIT_PACKAGE}/ HEAD | ssh $lxc_ip "tar xv"

    ssh $lxc_ip "sudo apt-get -y install python-pip"
    ssh $lxc_ip "sudo pip install sphinx==1.3.4"

    ssh $lxc_ip "cd ${GEM_GIT_PACKAGE} ; export PYTHONPATH=\$PWD ; cd doc ; make html"
    scp -r "$lxc_ip:${GEM_GIT_PACKAGE}/doc/build/html" "out_${BUILD_UBUVER}/"

    trap ERR

    return
}

#
#  deps_list <listtype> <filename> - retrieve dependencies list from debian/control and debian/rules
#                                    to be able to install them without the package
#      listtype    inform deps_list which control lines use to get dependencies
#      filename    control file used for input
#
deps_list() {
    local old_ifs out_list skip i d listtype="$1" control_file="$2"/control rules_file="$2"/rules

    rules_dep=$(grep "^${BUILD_UBUVER^^}_DEP *= *" $rules_file | sed 's/([^)]*)//g' | sed 's/^.*= *//g')
    rules_rec=$(grep "^${BUILD_UBUVER^^}_REC *= *" $rules_file | sed 's/([^)]*)//g' | sed 's/^.*= *//g')

    out_list=""
    if [ "$listtype" = "all" ]; then
        in_list="$((cat "$control_file" | egrep '^Depends:|^Recommends:|Build-Depends:' | sed 's/^\(Build-\)\?Depends://g;s/^Recommends://g' ; echo ", $rules_dep, $rules_rec") | tr '\n' ','| sed 's/,\+/,/g')"
    elif [  "$listtype" = "deprec" ]; then
        in_list="$((cat "$control_file" | egrep '^Depends:|^Recommends:' | sed 's/^Depends://g;s/^Recommends://g' ; echo ", $rules_dep, $rules_rec") | tr '\n' ','| sed 's/,\+/,/g')"
    elif [  "$listtype" = "build" ]; then
        in_list="$((cat "$control_file" | egrep '^Depends:|^Build-Depends:' | sed 's/^\(Build-\)\?Depends://g' ; echo ", $rules_dep") | tr '\n' ','| sed 's/,\+/,/g')"
    else
        in_list="$((cat "$control_file" | egrep "^Depends:" | sed 's/^Depends: //g'; echo ", $rules_dep") | tr '\n' ','| sed 's/,\+/,/g')"
    fi

    old_ifs="$IFS"
    IFS=','
    for i in $in_list ; do
        item="$(echo "$i" |  sed 's/^ \+//g;s/ \+$//g')"
        pkg_name="$(echo "${item} " | cut -d ' ' -f 1)"
        pkg_vers="$(echo "${item} " | cut -d ' ' -f 2)"
        echo "[$pkg_name][$pkg_vers]" >&2
        if echo "$pkg_name" | grep -q "^\${" ; then
            continue
        fi
        skip=0
        for d in $(echo "$GEM_GIT_DEPS" | sed 's/ /,/g'); do
            if [ "$pkg_name" = "python-${d}" ]; then
                skip=1
                break
            fi
        done
        if [ $skip -eq 1 ]; then
            continue
        fi

        if [ "$out_list" = "" ]; then
            out_list="$pkg_name"
        else
            out_list="$out_list $pkg_name"
        fi
    done
    IFS="$old_ifs"

    echo "$out_list"

    return 0
}


_lxc_name_and_ip_get()
{
    local filename="$1" i e

    i=-1
    e=-1
    for i in $(seq 1 40); do
        sleep 2
        if grep -q "sudo lxc-console -n $GEM_EPHEM_NAME" /tmp/packager.eph.$$.log 2>&1 ; then
            lxc_name="$(grep "sudo lxc-console -n $GEM_EPHEM_NAME" /tmp/packager.eph.$$.log | grep -v '+ echo' | sed "s/.*sudo lxc-console -n \($GEM_EPHEM_NAME\)/\1/g")"
            for e in $(seq 1 40); do
                sleep 2
                if grep -q "$lxc_name" /var/lib/misc/dnsmasq*.leases ; then
                    lxc_ip="$(grep " $lxc_name " /var/lib/misc/dnsmasq*.leases | tail -n 1 | cut -d ' ' -f 3)"
                    break
                fi
            done
            break
        fi
    done
    if [ $i -eq 40 -o $e -eq 40 ]; then
        return 1
    fi
    echo "SUCCESSFULLY RUNNED $lxc_name ($lxc_ip)"

    return 0
}

devtest_run () {
    if [ ! -d "out_${BUILD_UBUVER}" ]; then
        mkdir "out_${BUILD_UBUVER}"
    fi

    sudo echo
    sudo ${GEM_EPHEM_CMD} -o $GEM_EPHEM_NAME -d 2>&1 | tee /tmp/packager.eph.$$.log &
    _lxc_name_and_ip_get /tmp/packager.eph.$$.log

    _wait_ssh $lxc_ip

    set +e
    _devtest_innervm_run $lxc_ip
    inner_ret=$?
    sudo $LXC_TERM -n $lxc_name
    set -e

    if [ -f /tmp/packager.eph.$$.log ]; then
        rm /tmp/packager.eph.$$.log
    fi

    return $inner_ret
}


pkgtest_run () {
    local i e branch_id="$1" commit

    commit="$(git log --pretty='format:%h' -1)"

    #
    #  run build of package
    if [ -d build-deb ]; then
        if [ ! -f build-deb/${GEM_DEB_PACKAGE}_*.deb ]; then
            echo "'build-deb' directory already exists but .deb file package was not found"
            return 1

        fi
    else
        $0 $BUILD_FLAGS
    fi

    #
    #  prepare repo and install $GEM_DEB_PACKAGE package
    cd build-deb
    dpkg-scanpackages . /dev/null >Packages
    cat Packages | gzip -9c > Packages.gz
    dpkg-scansources . > Sources
    cat Sources | gzip > Sources.gz
    cat > Release <<EOF
Archive: $BUILD_UBUVER
Origin: Ubuntu
Label: Local Ubuntu ${BUILD_UBUVER^} Repository
Architecture: amd64
MD5Sum:
EOF
    printf ' '$(md5sum Packages | cut --delimiter=' ' --fields=1)' %16d Packages\n' \
        $(wc --bytes Packages | cut --delimiter=' ' --fields=1) >> Release
    printf ' '$(md5sum Packages.gz | cut --delimiter=' ' --fields=1)' %16d Packages.gz\n' \
        $(wc --bytes Packages.gz | cut --delimiter=' ' --fields=1) >> Release
    printf ' '$(md5sum Sources | cut --delimiter=' ' --fields=1)' %16d Sources\n' \
        $(wc --bytes Sources | cut --delimiter=' ' --fields=1) >> Release
    printf ' '$(md5sum Sources.gz | cut --delimiter=' ' --fields=1)' %16d Sources.gz\n' \
        $(wc --bytes Sources.gz | cut --delimiter=' ' --fields=1) >> Release
    gpg --armor --detach-sign --output Release.gpg Release
    cd -

    sudo echo
    sudo ${GEM_EPHEM_CMD} -o $GEM_EPHEM_NAME -d 2>&1 | tee /tmp/packager.eph.$$.log &
    _lxc_name_and_ip_get /tmp/packager.eph.$$.log

    _wait_ssh $lxc_ip

    set +e
    _pkgtest_innervm_run $lxc_ip
    inner_ret=$?
    sudo $LXC_TERM -n $lxc_name
    set -e

    if [ -f /tmp/packager.eph.$$.log ]; then
        rm /tmp/packager.eph.$$.log
    fi

    if [ $inner_ret -ne 0 ]; then
        return $inner_ret
    fi

    if [ $BUILD_REPOSITORY -eq 1 -a -d "${GEM_DEB_REPO}" ]; then
        if [ "$branch_id" != "" ]; then
            repo_id="$(repo_id_get)"
            if [ "git://$repo_id" != "$GEM_GIT_REPO" -o "$branch_id" != "master" ]; then
                CUSTOM_SERIE="devel/$(echo "$repo_id" | sed "s@/@__@g;s/\./-/g")__${branch_id}"
                if [ "$CUSTOM_SERIE" != "" ]; then
                    GEM_DEB_SERIE="$CUSTOM_SERIE"
                fi
            fi
        fi
        mkdir -p "${GEM_DEB_REPO}/${BUILD_UBUVER}/${GEM_DEB_SERIE}"
        repo_tmpdir="$(mktemp -d "${GEM_DEB_REPO}/${BUILD_UBUVER}/${GEM_DEB_SERIE}/${GEM_DEB_PACKAGE}.${commit}.XXXXXX")"

        # if the monotone directory exists and is the "gem" repo and is the "master" branch then ...
        if [ -d "${GEM_DEB_MONOTONE}/${BUILD_UBUVER}/binary" ]; then
            if [ "git://$repo_id" == "$GEM_GIT_REPO" -a "$branch_id" == "master" ]; then
                cp build-deb/${GEM_DEB_PACKAGE}_*.deb build-deb/${GEM_DEB_PACKAGE}_*.changes \
                    build-deb/${GEM_DEB_PACKAGE}_*.dsc build-deb/${GEM_DEB_PACKAGE}_*.tar.gz \
                    "${GEM_DEB_MONOTONE}/${BUILD_UBUVER}/binary"
            fi
        fi

        cp build-deb/${GEM_DEB_PACKAGE}_*.deb build-deb/${GEM_DEB_PACKAGE}_*.changes \
            build-deb/${GEM_DEB_PACKAGE}_*.dsc build-deb/${GEM_DEB_PACKAGE}_*.tar.gz \
            build-deb/Packages* build-deb/Sources* build-deb/Release* "${repo_tmpdir}"
        if [ "${GEM_DEB_REPO}/${BUILD_UBUVER}/${GEM_DEB_SERIE}/${GEM_DEB_PACKAGE}.${commit}" ]; then
            rm -rf "${GEM_DEB_REPO}/${BUILD_UBUVER}/${GEM_DEB_SERIE}/${GEM_DEB_PACKAGE}.${commit}"
        fi
        mv "${repo_tmpdir}" "${GEM_DEB_REPO}/${BUILD_UBUVER}/${GEM_DEB_SERIE}/${GEM_DEB_PACKAGE}.${commit}"
        echo "The package is saved here: ${GEM_DEB_REPO}/${BUILD_UBUVER}/${GEM_DEB_SERIE}/${GEM_DEB_PACKAGE}.${commit}"
    fi

    # TODO
    # app related tests (run demos)

    return
}

builddoc_run () {
    if [ ! -d "out_${BUILD_UBUVER}" ]; then
        mkdir "out_${BUILD_UBUVER}"
    fi

    sudo echo
    sudo ${GEM_EPHEM_CMD} -o $GEM_EPHEM_NAME -d 2>&1 | tee /tmp/packager.eph.$$.log &
    _lxc_name_and_ip_get /tmp/packager.eph.$$.log

    _wait_ssh $lxc_ip

    set +e
    _builddoc_innervm_run $lxc_ip
    inner_ret=$?
    sudo $LXC_TERM -n $lxc_name
    set -e

    if [ -f /tmp/packager.eph.$$.log ]; then
        rm /tmp/packager.eph.$$.log
    fi

    return $inner_ret
}

#
#  MAIN
#
BUILD_SOURCES_COPY=0
BUILD_BINARIES=0
BUILD_REPOSITORY=0
BUILD_DEVEL=0
BUILD_UNSIGN=0
BUILD_UBUVER_REFERENCE="precise"
BUILD_UBUVER="$BUILD_UBUVER_REFERENCE"
BUILD_ON_LXC=0
BUILD_FLAGS=""

trap sig_hand SIGINT SIGTERM
#  args management
while [ $# -gt 0 ]; do
    case $1 in
        -D|--development)
            BUILD_DEVEL=1
            if [ "$DEBFULLNAME" = "" -o "$DEBEMAIL" = "" ]; then
                echo
                echo "ERROR: set DEBFULLNAME and DEBEMAIL environment vars and run again the script"
                echo
                exit 1
            fi
            ;;
        -s|--serie)
            BUILD_UBUVER="$2"
            if [ "$BUILD_UBUVER" != "precise" -a "$BUILD_UBUVER" != "trusty" ]; then
                echo
                echo "ERROR: ubuntu version '$BUILD_UBUVER' not supported"
                echo
                exit 1
            fi
            BUILD_FLAGS="$BUILD_FLAGS $1"
            shift
            ;;
        -S|--sources_copy)
            BUILD_SOURCES_COPY=1
            ;;
        -B|--binaries)
            BUILD_BINARIES=1
            ;;
        -R|--repository)
            BUILD_REPOSITORY=1
            ;;
        -U|--unsigned)
            BUILD_UNSIGN=1
            ;;
        -L|--lxc_build)
            BUILD_ON_LXC=1
            ;;
        -h|--help)
            usage 0
            break
            ;;
        devtest)
            # Sed removes 'origin/' from the branch name
            devtest_run $(echo "$2" | sed 's@.*/@@g')
            exit $?
            break
            ;;
        pkgtest)
            # Sed removes 'origin/' from the branch name
            pkgtest_run $(echo "$2" | sed 's@.*/@@g')
            exit $?
            break
            ;;
        builddoc)
            builddoc_run $(echo "$2" | sed 's@.*/@@g')
            exit $?
            break
            ;;
        *)
            usage 1
            break
            ;;
    esac
    BUILD_FLAGS="$BUILD_FLAGS $1"
    shift
done

DPBP_FLAG=""
if [ $BUILD_BINARIES -eq 0 ]; then
    DPBP_FLAG="-S"
fi
if [ $BUILD_UNSIGN -eq 1 ]; then
    DPBP_FLAG="$DPBP_FLAG -us -uc"
fi

mksafedir "$GEM_BUILD_ROOT"
mksafedir "$GEM_BUILD_SRC"

git archive HEAD | (cd "$GEM_BUILD_SRC" ; tar xv)

# NOTE: if in the future we need modules we need to execute the following commands
#
# git submodule init
# git submodule update
##  "submodule foreach" vars: $name, $path, $sha1 and $toplevel:
# git submodule foreach "git archive HEAD | (cd \"\${toplevel}/${GEM_BUILD_SRC}/\$path\" ; tar xv ) "

# date
if [ -f gem_date_file ]; then
    dt="$(cat gem_date_file)"
else
    dt="$(date +%s)"
    echo "$dt" > gem_date_file
fi

cd "$GEM_BUILD_SRC"

# version info from openquake/hazardlib/__init__.py
ini_vers="$(cat openquake/hazardlib/__init__.py | sed -n "s/^__version__[  ]*=[    ]*['\"]\([^'\"]\+\)['\"].*/\1/gp")"
ini_maj="$(echo "$ini_vers" | sed -n 's/^\([0-9]\+\).*/\1/gp')"
ini_min="$(echo "$ini_vers" | sed -n 's/^[0-9]\+\.\([0-9]\+\).*/\1/gp')"
ini_bfx="$(echo "$ini_vers" | sed -n 's/^[0-9]\+\.[0-9]\+\.\([0-9]\+\).*/\1/gp')"
ini_suf="" # currently not included into the version array structure

# version info from debian/changelog
h="$(grep "^$GEM_DEB_PACKAGE" debian/changelog | head -n 1)"

# is it the first item of changelog ?
h_first="$(cat debian/changelog | head -n 1)"
h_is_first=0
if [ "$h" = "$h_first" ]; then
    h_is_first=1
fi

# pkg_vers="$(echo "$h" | cut -d ' ' -f 2 | cut -d '(' -f 2 | cut -d ')' -f 1 | sed -n 's/[-+].*//gp')"
pkg_name="$(echo "$h" | cut -d ' ' -f 1)"
pkg_vers="$(echo "$h" | cut -d ' ' -f 2 | cut -d '(' -f 2 | cut -d ')' -f 1)"
pkg_rest="$(echo "$h" | cut -d ' ' -f 3-)"
pkg_maj="$(echo "$pkg_vers" | sed -n 's/^\([0-9]\+\).*/\1/gp')"
pkg_min="$(echo "$pkg_vers" | sed -n 's/^[0-9]\+\.\([0-9]\+\).*/\1/gp')"
pkg_bfx="$(echo "$pkg_vers" | sed -n 's/^[0-9]\+\.[0-9]\+\.\([0-9]\+\).*/\1/gp')"
pkg_deb="$(echo "$pkg_vers" | sed -n 's/^[0-9]\+\.[0-9]\+\.[0-9]\+\(-[^+]\+\).*/\1/gp')"
pkg_suf="$(echo "$pkg_vers" | sed -n 's/^[0-9]\+\.[0-9]\+\.[0-9]\+-[^+]\+\(+.*\)/\1/gp')"
# echo "pkg [$pkg_vers] [$pkg_maj] [$pkg_min] [$pkg_bfx] [$pkg_deb] [$pkg_suf]"

if [ $BUILD_DEVEL -eq 1 ]; then
    commit="$(git log --pretty='format:%h' -1)"
    mv debian/changelog debian/changelog.orig

    if [ "$pkg_maj" = "$ini_maj" -a "$pkg_min" = "$ini_min" -a \
         "$pkg_bfx" = "$ini_bfx" -a "$pkg_deb" != "" ]; then
        deb_ct="$(echo "$pkg_deb" | sed 's/^-//g;s/~.*//g')"
        if [ $h_is_first -eq 1 ]; then
            pkg_deb="-$(( deb_ct ))"
        else
            pkg_deb="-$(( deb_ct + 1))"
        fi
    else
        pkg_maj="$ini_maj"
        pkg_min="$ini_min"
        pkg_bfx="$ini_bfx"
        pkg_deb="-0"
    fi

    (
      echo "$pkg_name (${pkg_maj}.${pkg_min}.${pkg_bfx}${pkg_deb}~${BUILD_UBUVER}01~dev${dt}+${commit}) ${BUILD_UBUVER}; urgency=low"
      echo
      echo "  [Automatic Script]"
      echo "  * Development version from $commit commit"
      echo
      cat debian/changelog.orig | sed -n "/^$GEM_DEB_PACKAGE/q;p"
      echo " -- $DEBFULLNAME <$DEBEMAIL>  $(date -d@$dt -R)"
      echo
    )  > debian/changelog
    cat debian/changelog.orig | sed -n "/^$GEM_DEB_PACKAGE/,\$ p" >> debian/changelog
    rm debian/changelog.orig
else
    cp debian/changelog debian/changelog.orig
    cat debian/changelog.orig | sed "1 s/${BUILD_UBUVER_REFERENCE}/${BUILD_UBUVER}/g" > debian/changelog
    rm debian/changelog.orig
fi

if [  "$ini_maj" != "$pkg_maj" -o \
      "$ini_min" != "$pkg_min" -o \
      "$ini_bfx" != "$pkg_bfx" ]; then
    echo
    echo "Versions are not aligned"
    echo "    init:  ${ini_maj}.${ini_min}.${ini_bfx}"
    echo "    pkg:   ${pkg_maj}.${pkg_min}.${pkg_bfx}"
    echo
    echo "press [enter] to continue, [ctrl+c] to abort"
    read a
fi

# the following unexecuted block of code is a flag to identify where and how modifications can
# be performed from sources to package
if [ 0 -eq 1 ]; then
    sed -i "s/^\([ 	]*\)[^)]*\()  # release date .*\)/\1${dt}\2/g" openquake/__init__.py

    # mods pre-packaging
    mv LICENSE         openquake
    mv README.rst      openquake/README
    mv celeryconfig.py openquake
    mv openquake.cfg   openquake

    mv bin/openquake   bin/oqscript.py
    mv bin             openquake/bin

    rm -rf $(find demos -mindepth 1 -maxdepth 1 | egrep -v 'demos/simple_fault_demo_hazard|demos/event_based_hazard|demos/_site_model')
fi

if [ $BUILD_ON_LXC -eq 1 ]; then
    sudo ${GEM_EPHEM_CMD} -o $GEM_EPHEM_NAME -d 2>&1 | tee /tmp/packager.eph.$$.log &
    _lxc_name_and_ip_get /tmp/packager.eph.$$.log
    _wait_ssh $lxc_ip

    set +e
    _pkgbuild_innervm_run $lxc_ip $DPBP_FLAG
    inner_ret=$?
    sudo $LXC_TERM -n $lxc_name
    set -e
    if [ -f /tmp/packager.eph.$$.log ]; then
        rm /tmp/packager.eph.$$.log
    fi
    if [ $inner_ret -ne 0 ]; then
        exit 1
    fi
else
    dpkg-buildpackage $DPBP_FLAG
fi
cd -

# if the monotone directory exists and is the "gem" repo and is the "master" branch then ...
if [ -d "${GEM_DEB_MONOTONE}/${BUILD_UBUVER}/source" -a $BUILD_SOURCES_COPY -eq 1 ]; then
    cp build-deb/${GEM_DEB_PACKAGE}_*.changes \
        build-deb/${GEM_DEB_PACKAGE}_*.dsc build-deb/${GEM_DEB_PACKAGE}_*.tar.gz \
        "${GEM_DEB_MONOTONE}/${BUILD_UBUVER}/source"
fi

if [ $BUILD_DEVEL -ne 1 ]; then
    exit 0
fi

#
# DEVEL EXTRACTION OF SOURCES
if [ -z "$GEM_SRC_PKG" ]; then
    echo "env var GEM_SRC_PKG not set, exit"
    exit 0
fi
GEM_BUILD_PKG="${GEM_SRC_PKG}/pkg"
mksafedir "$GEM_BUILD_PKG"
GEM_BUILD_EXTR="${GEM_SRC_PKG}/extr"
mksafedir "$GEM_BUILD_EXTR"
cp  ${GEM_BUILD_ROOT}/${GEM_DEB_PACKAGE}_*.deb  $GEM_BUILD_PKG
cd "$GEM_BUILD_EXTR"
dpkg -x $GEM_BUILD_PKG/${GEM_DEB_PACKAGE}_*.deb .
dpkg -e $GEM_BUILD_PKG/${GEM_DEB_PACKAGE}_*.deb
