using CourseApp1.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace CourseApp1.Models
{
    public class Trainee_CoursesModel
    {
        public int Trainee_id { get; set; }
        public int Course_id { get; set; }
        public System.DateTime Registration_date { get; set; }

        public virtual Cours Cours { get; set; }
        public virtual Trainee Trainee { get; set; }
    }
}