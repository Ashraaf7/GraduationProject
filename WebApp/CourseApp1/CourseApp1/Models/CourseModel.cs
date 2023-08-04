using CourseApp1.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace CourseApp1.Models
{
    public class CourseModel
    {
        public int ID { get; set; }
        public string Name { get; set; }
        public Nullable<System.DateTime> Creation_Date { get; set; }
        public string Description { get; set; }
        public int Category_id { get; set; }
        public Nullable<int> Trainer_id { get; set; }
        public string CourseLink { get; set; }
        public Nullable<bool> IsYoutube { get; set; }
        public string ImgPath { get; set; }
        public HttpPostedFileBase ImageFile { get; set; }
        public virtual Category Category { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<Course_lessons> Course_lessons { get; set; }
        public virtual Trainer Trainer { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<Trainee_Courses> Trainee_Courses { get; set; }

    }
}