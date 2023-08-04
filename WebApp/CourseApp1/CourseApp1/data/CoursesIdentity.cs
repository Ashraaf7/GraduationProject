using Microsoft.AspNet.Identity.EntityFramework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    public class CoursesIdentity:IdentityDbContext<MyIdentityUser>
    {
        public CoursesIdentity():base("Courses_Connection")
        {

        }
    }
    public class MyIdentityUser : IdentityUser
    {

    }
}